"""
Builds a public, real-world evaluation set from Banking77.

Banking77 (PolyAI, Casanueva et al. 2020, CC BY 4.0) is ~13k real customer
queries to a banking support bot, each labeled with one of 77 intents.
Source: https://github.com/PolyAI-LDN/task-specific-datasets

77 intents is far finer than a support desk's routing queues, so the intents
are grouped below into 9 queues. THE GROUPING IS MY OWN DESIGN CHOICE, not part
of the dataset, so any accuracy number should be quoted as "on Banking77 grouped
into 9 queues", never as a Banking77 benchmark result.

Output: data/public/banking77_sample.json
    pool: 8 train examples per intent (616) used as the retrieval pool
    eval: 200 held-out test examples (disjoint from the pool)

Run: python data/fetch_public_dataset.py
"""

import csv
import io
import json
import random
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data"
OUT = Path(__file__).parent / "public" / "banking77_sample.json"
SEED = 7
POOL_PER_INTENT = 8
EVAL_SIZE = 200

GROUPS = {
    "Card Delivery & Activation": [
        "activate_my_card", "card_arrival", "card_delivery_estimate", "card_linking",
        "order_physical_card", "get_physical_card", "getting_spare_card",
        "getting_virtual_card", "get_disposable_virtual_card", "disposable_card_limits",
        "card_about_to_expire",
    ],
    "Card Usage Problems": [
        "card_not_working", "contactless_not_working", "virtual_card_not_working",
        "card_swallowed", "pin_blocked", "change_pin", "card_acceptance",
        "visa_or_mastercard", "supported_cards_and_currencies", "country_support",
        "apple_pay_or_google_pay",
    ],
    "Fraud & Lost Cards": [
        "lost_or_stolen_card", "lost_or_stolen_phone", "compromised_card",
        "card_payment_not_recognised", "cash_withdrawal_not_recognised",
        "direct_debit_payment_not_recognised",
    ],
    "Transfers": [
        "balance_not_updated_after_bank_transfer",
        "balance_not_updated_after_cheque_or_cash_deposit", "beneficiary_not_allowed",
        "cancel_transfer", "declined_transfer", "failed_transfer", "pending_transfer",
        "receiving_money", "transfer_into_account", "transfer_not_received_by_recipient",
        "transfer_timing",
    ],
    "Top-ups": [
        "automatic_top_up", "pending_top_up", "top_up_by_bank_transfer_charge",
        "top_up_by_card_charge", "top_up_by_cash_or_cheque", "top_up_failed",
        "top_up_limits", "top_up_reverted", "topping_up_by_card", "verify_top_up",
    ],
    "Card Payments & Refunds": [
        "declined_card_payment", "pending_card_payment", "card_payment_fee_charged",
        "card_payment_wrong_exchange_rate", "transaction_charged_twice",
        "reverted_card_payment?", "Refund_not_showing_up", "request_refund",
        "extra_charge_on_statement",
    ],
    "Cash Withdrawals": [
        "atm_support", "cash_withdrawal_charge", "declined_cash_withdrawal",
        "pending_cash_withdrawal", "wrong_amount_of_cash_received",
        "wrong_exchange_rate_for_cash_withdrawal",
    ],
    "Exchange Rates & Fees": [
        "exchange_charge", "exchange_rate", "exchange_via_app", "fiat_currency_support",
        "transfer_fee_charged",
    ],
    "Identity & Account": [
        "verify_my_identity", "unable_to_verify_identity", "why_verify_identity",
        "verify_source_of_funds", "edit_personal_details", "passcode_forgotten",
        "terminate_account", "age_limit",
    ],
}
INTENT_TO_GROUP = {intent: group for group, intents in GROUPS.items() for intent in intents}


def _download(name: str) -> list[dict]:
    with urllib.request.urlopen(f"{BASE}/{name}", timeout=60) as resp:
        text = resp.read().decode("utf-8")
    return list(csv.DictReader(io.StringIO(text)))


def _item(prefix: str, i: int, row: dict, status: str) -> dict:
    return {
        "id": f"{prefix}-{i:04d}",
        "subject": "",
        "body": row["text"].strip(),
        "sender": "Customer",
        "sender_type": "customer",
        "intent": row["category"],
        "category": INTENT_TO_GROUP[row["category"]],
        "status": status,
    }


def build(train: list[dict], test: list[dict]) -> dict:
    unmapped = {r["category"] for r in train + test} - set(INTENT_TO_GROUP)
    assert not unmapped, f"Unmapped intents: {sorted(unmapped)}"
    rng = random.Random(SEED)

    by_intent: dict[str, list[dict]] = {}
    for r in train:
        by_intent.setdefault(r["category"], []).append(r)
    pool_rows = []
    for intent in sorted(by_intent):
        pool_rows += rng.sample(by_intent[intent], POOL_PER_INTENT)
    pool_texts = {r["text"].strip().lower() for r in pool_rows}

    test_by_intent: dict[str, list[dict]] = {}
    for r in test:
        if r["text"].strip().lower() not in pool_texts:
            test_by_intent.setdefault(r["category"], []).append(r)
    eval_rows = []
    for intent in sorted(test_by_intent):
        eval_rows += rng.sample(test_by_intent[intent], 2)
    remaining = [r for rows in test_by_intent.values() for r in rows if r not in eval_rows]
    eval_rows += rng.sample(remaining, EVAL_SIZE - len(eval_rows))
    rng.shuffle(eval_rows)

    return {
        "source": "Banking77 (PolyAI), https://github.com/PolyAI-LDN/task-specific-datasets",
        "license": "CC BY 4.0",
        "citation": "Casanueva et al. 2020, Efficient Intent Detection with Dual Sentence Encoders",
        "note": "77 intents grouped into 9 routing queues by the project author (not part of the dataset).",
        "categories": list(GROUPS),
        "pool": [_item("BP", i, r, "closed") for i, r in enumerate(pool_rows)],
        "eval": [_item("BE", i, r, "open") for i, r in enumerate(eval_rows)],
    }


def main():
    data = build(_download("train.csv"), _download("test.csv"))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, indent=1), encoding="utf-8")
    print(f"Wrote {OUT}: pool={len(data['pool'])} eval={len(data['eval'])} categories={len(data['categories'])}")


if __name__ == "__main__":
    main()
