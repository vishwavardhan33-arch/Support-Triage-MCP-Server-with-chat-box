"""
Generates the mock home-loan support dataset: data/tickets.json (240 tickets).

Design notes (these matter for evaluation honesty):
  * Subjects NEVER contain the category name. An earlier version put the label in
    the subject ("Disbursement Delay: BHL-123"), which let any model read the answer.
  * Each category has 6-7 body templates plus urgency / "no rush" modifiers, ~10%
    mixed-issue tickets (hard=True) and ~15% noisy, lower-cased, typo'd tickets.
  * Every ticket carries one identifier (BHL-xxxxx loan ref or PRT-xxxx partner code).
  * Labeled duplicate cases for evaluating duplicate detection:
      - 16 "same id, paraphrased" follow-ups        (duplicate_of = original id)
      - 8  "same sender, no id" follow-ups          (duplicate_of = original id)
      - 16 hard negatives: same id + sender but a DIFFERENT issue (duplicate_of absent)
  * `category` / `priority` are the ground-truth labels used by evaluate.py.

Run: python data/generate_data.py
"""

import json
import random
from datetime import datetime, timedelta
from pathlib import Path

SEED = 2026
N_BASE = 200
N_ID_DUPS = 16
N_FOLLOWUP_DUPS = 8
N_HARD_NEGATIVES = 16
BASE_DATE = datetime(2026, 10, 20, 10, 0, 0)

# category -> (id_kind, base_priority, subjects, bodies, followup)
SPEC = {
    "Loan Status Query": ("loan", "low",
        ["Status of application {ref}", "Any update on {ref}?", "Where is loan file {ref} stuck", "Checking progress on {ref}"],
        ["Any update on loan application {ref}? It's been {days} days since submission.",
         "Customer keeps calling for the sanction status of {ref}. Can you tell me which stage it is in?",
         "Submitted all docs for {ref} {days} days ago. What is the current stage and when can we expect the sanction letter?",
         "Need a quick status on {ref} before EOD, the customer is following up with me.",
         "Could you let me know whether {ref} has moved to credit approval or is still with the sales team?",
         "My home loan application {ref} shows 'under process' on the app for {days} days now. What does that mean?",
         "Partner is asking for a tentative sanction date on {ref}, the customer wants to finalize the property booking."],
        "Still waiting for an update on my application, please revert."),
    "Document Upload Issue": ("loan", "medium",
        ["Cannot upload salary slip for {ref}", "Upload failing on portal for {ref}", "Document rejected again, {ref}", "Portal not accepting my PDF"],
        ["Unable to upload the salary slip for {ref}, the portal shows a generic error.",
         "The upload keeps failing for Aadhaar and PAN on application {ref}, tried three browsers.",
         "Getting a 'file too large' error while uploading six months of bank statements for {ref}.",
         "Documents for {ref} were marked unreadable twice even though the scans are clear. Which format do you want?",
         "The upload button just spins when I attach the property papers for {ref}.",
         "The portal accepts the PDF but it vanishes after refresh, so the document list for {ref} is still empty."],
        "The upload problem is still there, nothing has changed on my side."),
    "Disbursement Delay": ("loan", "high",
        ["Funds not received for {ref}", "Payment pending since {days} days", "Money still not credited, {ref}", "Builder waiting for payment on {ref}"],
        ["Disbursement for {ref} was approved {days} days ago but the funds haven't reached the customer.",
         "Sanctioned amount of {amount} on {ref} is still not credited, the builder is demanding payment.",
         "Partner is escalating the pending payment to the seller on {ref}; the customer may cancel the booking.",
         "Final tranche for {ref} shows as 'initiated' but nothing has hit the account after {days} days.",
         "We are past the registry date and the tranche for {ref} is still pending. Please expedite.",
         "Customer says the bank confirmed no credit was received against {ref}. Where did the transfer go?"],
        "Following up again, the money has still not reached the account."),
    "Interest Rate Query": ("loan", "low",
        ["Rate difference on {ref}", "Which interest slab applies to {ref}?", "Why is my ROI higher than quoted"],
        ["Customer wants clarification on the interest rate slab applied to {ref}.",
         "The rate on the sanction letter for {ref} is higher than what was quoted on the app. Why?",
         "Can you explain whether the floating rate on {ref} is linked to repo and how often it resets?",
         "Partner is asking if a rate reduction is possible for {ref} given the customer's credit score.",
         "Is the processing fee included in the effective rate shown for {ref}?",
         "Customer compared our rate with a competitor and wants a revised offer on {ref}."],
        "Any response on my question about the interest rate?"),
    "Account Access": ("partner", "medium",
        ["Can't log in to dashboard", "OTP not coming", "Account locked, code {ref}", "Login keeps failing"],
        ["Partner {ref} is unable to log in to the dashboard, the OTP is not being received.",
         "Getting an 'account locked' message when trying to access the partner portal with code {ref}.",
         "The password reset email for {ref} never arrives, checked spam as well.",
         "Our team member's login under {ref} says invalid credentials even after a reset.",
         "The portal logs me out every few minutes, partner code {ref}, I cannot finish any application.",
         "Changed my registered mobile number but OTPs for {ref} still go to the old one."],
        "Still cannot log in, please help."),
    "Partner Onboarding": ("partner", "medium",
        ["Onboarding stuck for {days} days", "Agreement signed, no access yet", "Waiting for partner activation", "New DSA setup pending, {ref}"],
        ["New DSA partner {ref} onboarding is stuck at the verification step for {days} days.",
         "The partner agreement for {ref} was signed last week but portal access is not yet provisioned.",
         "How long does activation take after submitting the empanelment documents for {ref}?",
         "We completed the training module but the status of {ref} still shows 'pending approval'.",
         "Our firm wants to add two more users under {ref}. What is the process?",
         "Welcome kit and login details for {ref} have not been shared after onboarding."],
        "Any update on my onboarding? Still no access."),
    "KYC Issue": ("loan", "medium",
        ["KYC mismatch on {ref}", "Name differs between PAN and application", "Re-KYC pending", "Video KYC failed"],
        ["KYC mismatch flagged for {ref}, the name on PAN doesn't match the application.",
         "The customer's re-verification is pending and it is blocking progress on {ref}.",
         "Video KYC failed twice for {ref}, the agent said the face was not clear.",
         "The address on the Aadhaar is different from the current address on {ref}. Which proof should we upload?",
         "The CKYC record for {ref} shows incomplete although the customer submitted all papers.",
         "Date of birth in the bureau record doesn't match the application for {ref} and the case is on hold."],
        "KYC is still showing pending, can someone look at this?"),
    "Payment Reconciliation": ("partner", "medium",
        ["Payout short for partner {ref}", "Commission sheet mismatch", "Wrong payout amount last month", "TDS deducted twice, {ref}"],
        ["The payout for partner {ref} doesn't match the commission sheet, please reconcile.",
         "Last month's payout for {ref} is short by a fixed processing fee and needs review.",
         "TDS seems to have been deducted twice on the payout for {ref}.",
         "Two cases disbursed in September are missing from the payout statement for {ref}.",
         "The payout percentage applied for {ref} is lower than the grid we agreed on.",
         "Please share the breakup of deductions on the latest payout to {ref}."],
        "No reply on the payout difference I raised earlier."),
    "Technical / Portal Bug": ("loan", "low",
        ["EMI calculator showing wrong number", "Dashboard not loading", "Page error on {ref}", "App crashes on submit"],
        ["The loan calculator on the portal shows an incorrect EMI for {ref}.",
         "The dashboard is not loading application {ref}, it keeps spinning indefinitely.",
         "The app crashes every time I tap submit on the details page for {ref}.",
         "The status timeline for {ref} shows dates in the future.",
         "The download sanction letter button returns a 404 for {ref}.",
         "Search by application number returns nothing for {ref} although the file exists."],
        "The portal issue I reported is still happening."),
    "Complaint - Escalation": ("loan", "high",
        ["Very unhappy, need callback today", "Escalating repeated delays on {ref}", "Formal complaint about service", "No response for {days} days"],
        ["Customer is extremely unhappy with the delay on {ref} and wants a callback today.",
         "Partner is threatening to stop referring leads because of repeated delays on cases like {ref}.",
         "This is my {days}th follow-up on {ref} with no resolution. I am raising a formal complaint.",
         "Nobody has responded to the earlier tickets on {ref}. Please escalate to a senior manager.",
         "I will approach the banking ombudsman if {ref} is not resolved this week.",
         "Rude behaviour from the support agent on the call regarding {ref}. This is unacceptable."],
        "I have not heard back yet. Please treat this as urgent."),
}
CATEGORIES = list(SPEC)

URGENT = ["This is urgent.", "Needs attention today.", "The customer has a registry deadline this Friday.",
          "Escalating as this is blocking the customer's possession."]
NO_RUSH = ["No rush on this.", "Just checking, not urgent.", "Whenever you get a chance."]
PRIORITIES = ["low", "medium", "high"]
STATUSES = ["open", "in_progress", "closed"]
AMOUNTS = ["Rs 8 lakh", "Rs 12 lakh", "Rs 18 lakh", "Rs 25 lakh", "Rs 35 lakh", "Rs 50 lakh"]
FIRST = ["Rohan", "Priya", "Amit", "Sneha", "Vikram", "Anjali", "Karan", "Neha", "Arjun", "Divya", "Suresh", "Meera",
         "Rahul", "Pooja", "Nikhil", "Kavita", "Manish", "Ritu", "Sanjay", "Isha", "Deepak", "Swati", "Gaurav", "Nisha",
         "Varun", "Tanvi", "Harsh", "Aditi", "Mohit", "Shreya", "Ankit", "Komal", "Yash", "Preeti", "Dhruv", "Simran",
         "Abhishek", "Lakshmi", "Faisal", "Zoya"]
LAST = ["Sharma", "Verma", "Iyer", "Reddy", "Nair", "Gupta", "Joshi", "Rao", "Singh", "Mehta", "Kapoor", "Bose", "Menon",
        "Patel", "Das", "Chopra", "Malhotra", "Pillai", "Saxena", "Bhatt", "Khan", "Agarwal", "Sinha", "Dutta", "Ghosh"]


def typo(text: str, rng: random.Random) -> str:
    words = text.split(" ")
    idx = [i for i, w in enumerate(words) if len(w) > 4 and w.isalpha()]
    for i in rng.sample(idx, min(2, len(idx))):
        j = rng.randrange(1, len(words[i]) - 1)
        words[i] = words[i][:j] + words[i][j + 1:]
    return " ".join(words).lower() + rng.choice(["", " pls", " asap", " pls help"])


def shift(priority: str, delta: int) -> str:
    return PRIORITIES[max(0, min(2, PRIORITIES.index(priority) + delta))]


def make_id(kind: str, rng: random.Random, used: set) -> str:
    while True:
        ref = f"BHL-{rng.randint(10000, 99999)}" if kind == "loan" else f"PRT-{rng.randint(1000, 9999)}"
        if ref not in used:
            used.add(ref)
            return ref


def render(category: str, tpl_idx: int, ref: str, rng: random.Random) -> tuple[str, str]:
    _, _, subjects, bodies, _ = SPEC[category]
    fields = {"ref": ref, "days": rng.randint(2, 14), "amount": rng.choice(AMOUNTS)}
    return rng.choice(subjects).format(**fields), bodies[tpl_idx].format(**fields)


def main():
    rng = random.Random(SEED)
    names = [f"{first} {last}" for first in FIRST for last in LAST]
    rng.shuffle(names)
    used_refs: set = set()
    tickets: list[dict] = []
    tpl_of: dict[str, int] = {}

    def new_id() -> str:
        return f"T-{1000 + len(tickets)}"

    # ---- base tickets ----
    for i in range(N_BASE):
        category = CATEGORIES[i % len(CATEGORIES)]
        kind, base_priority, _, bodies, _ = SPEC[category]
        ref = make_id(kind, rng, used_refs)
        tpl_idx = rng.randrange(len(bodies))
        subject, body = render(category, tpl_idx, ref, rng)
        priority, hard = base_priority, False

        roll = rng.random()
        if roll < 0.20:
            body += " " + rng.choice(URGENT)
            priority = shift(priority, +1)
        elif roll < 0.30:
            body += " " + rng.choice(NO_RUSH)
            priority = shift(priority, -1)
        if rng.random() < 0.10:  # mixed-issue ticket: a second problem is mentioned, label stays the primary issue
            other = rng.choice([c for c in CATEGORIES if c != category and SPEC[c][0] == kind])
            extra = SPEC[other][3][rng.randrange(len(SPEC[other][3]))].format(
                ref=ref, days=rng.randint(2, 14), amount=rng.choice(AMOUNTS))
            body += " Also, " + extra[0].lower() + extra[1:]
            hard = True
        if rng.random() < 0.15:
            subject, body = subject.lower(), typo(body, rng)

        tid = new_id()
        tpl_of[tid] = tpl_idx
        tickets.append({
            "id": tid, "subject": subject, "body": body, "sender": names.pop(),
            "sender_type": "partner" if kind == "partner" or rng.random() < 0.3 else "customer",
            "reference_id": ref,
            "timestamp": (BASE_DATE - timedelta(minutes=rng.randint(0, 60 * 24 * 30))).strftime("%Y-%m-%dT%H:%M:%S"),
            "category": category, "priority": priority,
            "status": rng.choices(STATUSES, weights=[0.4, 0.3, 0.3])[0],
            "hard": hard, "source": "seed",
        })
    base = list(tickets)

    def later(ts: str) -> str:
        t = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S") + timedelta(hours=rng.randint(3, 72))
        return min(t, BASE_DATE).strftime("%Y-%m-%dT%H:%M:%S")

    # ---- labeled duplicates (same id, paraphrased) ----
    for orig in rng.sample(base, N_ID_DUPS):
        cat, ref = orig["category"], orig["reference_id"]
        choices = [i for i in range(len(SPEC[cat][3])) if i != tpl_of[orig["id"]]]
        subject, body = render(cat, rng.choice(choices), ref, rng)
        tickets.append({**orig, "id": new_id(), "subject": "Re: " + subject,
                        "body": rng.choice(["Following up: ", "Second request. ", ""]) + body,
                        "timestamp": later(orig["timestamp"]), "status": "open", "hard": False,
                        "duplicate_of": orig["id"]})

    # ---- labeled duplicates (same sender, no id) ----
    for orig in rng.sample(base, N_FOLLOWUP_DUPS):
        tickets.append({**orig, "id": new_id(), "subject": rng.choice(["Following up", "Any update?", "Re: my earlier request"]),
                        "body": SPEC[orig["category"]][4], "reference_id": None, "timestamp": later(orig["timestamp"]),
                        "status": "open", "hard": False, "duplicate_of": orig["id"]})

    # ---- hard negatives: same id + sender, different issue, NOT duplicates ----
    for orig in rng.sample([t for t in base if SPEC[t["category"]][0] == "loan"], N_HARD_NEGATIVES):
        cat = rng.choice([c for c in CATEGORIES if c != orig["category"] and SPEC[c][0] == "loan"])
        subject, body = render(cat, rng.randrange(len(SPEC[cat][3])), orig["reference_id"], rng)
        tickets.append({**orig, "id": new_id(), "subject": subject, "body": body, "category": cat,
                        "priority": SPEC[cat][1], "timestamp": later(orig["timestamp"]), "status": "open", "hard": False})

    tickets.sort(key=lambda t: t["timestamp"], reverse=True)
    out = Path(__file__).parent / "tickets.json"
    out.write_text(json.dumps(tickets, indent=2), encoding="utf-8")
    refs = sum(1 for t in tickets if t["reference_id"])
    print(f"Wrote {len(tickets)} tickets to {out} ({refs} with ids, "
          f"{sum('duplicate_of' in t for t in tickets)} labeled duplicates, {sum(t['hard'] for t in tickets)} hard)")


if __name__ == "__main__":
    main()
