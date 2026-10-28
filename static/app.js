const API = "/api";
const CATEGORIES = [
  "Loan Status Query", "Document Upload Issue", "Disbursement Delay",
  "Interest Rate Query", "Account Access", "Partner Onboarding",
  "KYC Issue", "Payment Reconciliation", "Technical / Portal Bug",
  "Complaint - Escalation", "Unclassified",
];

let tickets = [];
let currentStatus = "";
let currentCategory = "";
let selectedId = null;
let exampleIndex = 0;

const $ = (id) => document.getElementById(id);
const statusTabs = $("statusTabs");
const categoryFilter = $("categoryFilter");
const ticketRows = $("ticketRows");
const emptyState = $("emptyState");
const ticketCount = $("ticketCount");
const casePanelInner = $("casePanelInner");
const modelPlate = $("modelPlate");
const modelLabel = $("modelLabel");
const askForm = $("askForm");
const askInput = $("askInput");
const askAnswer = $("askAnswer");

// Every piece of ticket text is user-controlled, so it is always escaped before it touches innerHTML.
function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function formatDate(iso) {
  const d = new Date(iso);
  if (isNaN(d)) return "";
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric" }) + ", " +
    d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });
}

const priorityLabel = (p) => (p === "unclassified" ? "unrated" : p || "unrated");
const statusLabel = (s) => String(s || "").replace("_", " ");

async function errorText(res, fallback) {
  try {
    const body = await res.json();
    if (typeof body.detail === "string") return body.detail;
    if (Array.isArray(body.detail) && body.detail.length) {
      return "Please check the form: " + body.detail.map((d) => `${(d.loc || []).slice(-1)[0]} ${d.msg}`).join("; ");
    }
  } catch { /* fall through */ }
  return fallback;
}

function init() {
  CATEGORIES.forEach((c) => {
    const opt = document.createElement("option");
    opt.value = c;
    opt.textContent = c;
    categoryFilter.appendChild(opt);
  });

  statusTabs.addEventListener("click", (e) => {
    const btn = e.target.closest(".tab");
    if (!btn) return;
    document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
    btn.classList.add("active");
    currentStatus = btn.dataset.status;
    loadTickets();
  });
  categoryFilter.addEventListener("change", () => { currentCategory = categoryFilter.value; loadTickets(); });
  $("newCaseBtn").addEventListener("click", openNewCaseForm);
  askForm.addEventListener("submit", submitAsk);

  loadTickets();
  checkHealth();
  setInterval(checkHealth, 30000);
}

async function checkHealth() {
  try {
    const res = await fetch(`${API}/health`);
    const data = await res.json();
    const llm = data.llm || {};
    if (llm.configured) {
      modelPlate.className = "model-plate up";
      modelLabel.innerHTML = `Model: ${esc(llm.model)}<span class="plate-sub">Search: ${esc(data.embeddings)} on ${esc(data.vector_store)}</span>`;
    } else {
      modelPlate.className = "model-plate down";
      modelLabel.innerHTML = `Model not configured<span class="plate-sub">Set GROQ_API_KEY on the server</span>`;
    }
  } catch {
    modelPlate.className = "model-plate down";
    modelLabel.textContent = "Backend unreachable";
  }
}

async function loadTickets() {
  const params = new URLSearchParams();
  if (currentStatus) params.set("status", currentStatus);
  if (currentCategory) params.set("category", currentCategory);
  const res = await fetch(`${API}/tickets?${params}`);
  tickets = res.ok ? await res.json() : [];
  renderTable();
}

function renderTable() {
  ticketRows.innerHTML = "";
  ticketCount.textContent = tickets.length;
  emptyState.hidden = tickets.length > 0;

  tickets.forEach((t) => {
    const pending = t.category === "Unclassified";
    const tr = document.createElement("tr");
    if (t.id === selectedId) tr.classList.add("selected");
    tr.innerHTML = `
      <td class="case-id">${esc(t.id)}</td>
      <td>
        <div class="subject-main">${esc(t.subject) || "(no subject)"}</div>
        <div class="subject-cat ${pending ? "pending" : ""}">${pending ? "Not triaged yet" : esc(t.category)}</div>
      </td>
      <td>${esc(t.sender)}</td>
      <td><span class="stamp ${esc(t.priority)}">${esc(priorityLabel(t.priority))}</span></td>
      <td><span class="badge-status ${esc(t.status)}">${esc(statusLabel(t.status))}</span></td>
      <td class="received-cell">${esc(formatDate(t.timestamp))}</td>
    `;
    tr.addEventListener("click", () => openCase(t.id));
    ticketRows.appendChild(tr);
  });
}

// ------------------------------------------------------------------ case file
async function openCase(id) {
  selectedId = id;
  renderTable();
  const res = await fetch(`${API}/tickets/${encodeURIComponent(id)}`);
  if (!res.ok) return;
  const ticket = await res.json();
  renderCase(ticket);
  loadDuplicates(ticket.id);
}

async function loadDuplicates(id) {
  try {
    const res = await fetch(`${API}/tickets/${encodeURIComponent(id)}/duplicates`);
    if (res.ok && selectedId === id) renderDuplicates((await res.json()).possible_duplicates);
  } catch { /* the notice is a nicety, never block the case file on it */ }
}

function renderDuplicates(list) {
  const slot = $("dupNotice");
  if (!slot) return;
  if (!list || !list.length) { slot.innerHTML = ""; return; }
  slot.innerHTML = `
    <div class="notice">
      <p class="notice-title">Possible duplicate of ${list.length === 1 ? "another case" : `${list.length} other cases`}</p>
      <ul>
        ${list.map((d) => `
          <li><button type="button" class="link-btn" data-open="${esc(d.id)}">${esc(d.id)}</button>,
            ${esc(statusLabel(d.status))}: ${esc(d.subject)}<br>
            <span class="dim">${esc(d.reason)} (similarity ${esc(d.similarity)})</span></li>`).join("")}
      </ul>
    </div>`;
  slot.querySelectorAll("[data-open]").forEach((b) => b.addEventListener("click", () => openCase(b.dataset.open)));
}

function renderCase(ticket) {
  const pending = ticket.category === "Unclassified";
  const seeded = ticket.source !== "ui";
  casePanelInner.innerHTML = `
    <p class="case-heading">Submitted by ${esc(ticket.sender)} (${esc(ticket.sender_type)}) on ${esc(formatDate(ticket.timestamp))}</p>
    <h2 class="case-subject">${esc(ticket.subject) || "(no subject)"}</h2>
    <div class="case-meta">
      <span class="stamp ${esc(ticket.priority)}">${esc(priorityLabel(ticket.priority))}</span>
      <span class="badge-status ${esc(ticket.status)}">${esc(statusLabel(ticket.status))}</span>
    </div>
    <p class="case-category">${pending ? "Not triaged yet" : `Filed under <strong>${esc(ticket.category)}</strong>`}</p>
    <div id="dupNotice"></div>
    <div class="case-body">${esc(ticket.body)}</div>

    <div class="case-actions">
      <select class="status-select" id="statusSelect">
        ${["open", "in_progress", "closed"].map((s) => `<option value="${s}" ${s === ticket.status ? "selected" : ""}>${statusLabel(s)}</option>`).join("")}
      </select>
      <button class="primary" id="saveStatusBtn">Update status</button>
    </div>

    <div class="case-actions">
      <button class="primary" id="classifyBtn">Run AI triage</button>
    </div>
    ${seeded ? `<p class="form-hint">This is a seeded case, so its labels stay fixed. Triage shows what the model would predict.</p>` : ""}
    <div id="classifyResult"></div>
  `;

  $("saveStatusBtn").addEventListener("click", async () => {
    const res = await fetch(`${API}/tickets/${encodeURIComponent(ticket.id)}/status`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ status: $("statusSelect").value }),
    });
    if (res.ok) {
      await loadTickets();
      renderCase(await res.json());
      loadDuplicates(ticket.id);
    }
  });
  $("classifyBtn").addEventListener("click", () => runClassify(ticket));

  if (ticket.triage) {
    renderClassification({ category: ticket.category, category_valid: true, priority: ticket.priority, ...ticket.triage });
  }
}

// ------------------------------------------------------------------ triage
async function runClassify(ticket) {
  const resultDiv = $("classifyResult");
  const btn = $("classifyBtn");
  btn.disabled = true;
  resultDiv.innerHTML = `<p class="spinner-text">Thinking this one through&hellip;</p>`;
  const apply = ticket.source === "ui";

  try {
    const res = await fetch(`${API}/tickets/${encodeURIComponent(ticket.id)}/classify?apply=${apply}`, { method: "POST" });
    if (!res.ok) {
      resultDiv.innerHTML = `<div class="error-box">${esc(await errorText(res, "Classification failed."))}</div>`;
      return;
    }
    const data = await res.json();
    if (data.applied && data.ticket) {
      await loadTickets();
      if (selectedId === ticket.id) {
        renderCase(data.ticket);
        loadDuplicates(ticket.id);
        renderClassification(data);
      }
    } else if (selectedId === ticket.id) {
      renderClassification(data);
    }
  } catch {
    resultDiv.innerHTML = `<div class="error-box">Could not reach the backend.</div>`;
  } finally {
    const again = $("classifyBtn");
    if (again) again.disabled = false;
  }
}

function renderClassification(data) {
  const resultDiv = $("classifyResult");
  if (!resultDiv) return;
  const similar = data.similar_ticket_ids || [];
  resultDiv.innerHTML = `
    <div class="result-block">
      <div class="result-row">
        <span class="result-label">Predicted category</span>
        <strong>${esc(data.category || "\u2014")}${data.category_valid === false ? " (not in the category list)" : ""}</strong>
      </div>
      <div class="result-row">
        <span class="result-label">Predicted priority</span>
        <span class="stamp ${esc(data.priority || "unclassified")}">${esc(priorityLabel(data.priority))}</span>
      </div>
      <div class="result-row">
        <span class="result-label">Sentiment</span>
        <span>${esc(data.sentiment || "\u2014")}</span>
      </div>
      <p class="result-label">Suggested response</p>
      <div class="response-box">${esc(data.suggested_response || "\u2014")}</div>
      <button class="copy-btn" id="copyBtn">Copy response</button>
      ${similar.length ? `<p class="similar-list">Referenced similar cases: ${similar.map(esc).join(", ")}</p>` : ""}
      ${data.model ? `<p class="similar-list">Model: ${esc(data.model)}</p>` : ""}
    </div>`;
  $("copyBtn").addEventListener("click", () => navigator.clipboard && navigator.clipboard.writeText(data.suggested_response || ""));
}

// ------------------------------------------------------------------ new case
function examples() {
  const seeded = tickets.find((t) => t.source !== "ui" && t.reference_id && !t.duplicate_of);
  const list = [
    { sender: "Rohit Menon", sender_type: "partner", subject: "Funds still not received for BHL-48213",
      body: "The disbursement for BHL-48213 was approved 6 days ago but the customer has not received the money. The builder is asking for payment and the customer may cancel the booking. Please expedite." },
    { sender: "Anita Desai", sender_type: "customer", subject: "Cannot upload my salary slip",
      body: "I keep getting an error whenever I upload my latest salary slip for my home loan application. Tried Chrome and Edge. Pls help." },
  ];
  if (seeded) {
    list.push({ sender: seeded.sender, sender_type: seeded.sender_type, subject: "Re: " + seeded.subject,
      body: "Following up: " + seeded.body });
  }
  return list;
}

function openNewCaseForm() {
  selectedId = null;
  renderTable();
  casePanelInner.innerHTML = `
    <h2 class="case-subject">File a new case</h2>
    <p class="form-hint">Write it like a real message. Triage runs as soon as you file it.</p>
    <form id="newCaseForm">
      <div class="field"><label for="ncSender">Name</label>
        <input id="ncSender" maxlength="80" placeholder="Priya Sharma" autocomplete="off"></div>
      <div class="field"><label for="ncType">Writing as</label>
        <select id="ncType"><option value="customer">Customer</option><option value="partner">Partner</option></select></div>
      <div class="field"><label for="ncSubject">Subject</label>
        <input id="ncSubject" maxlength="200" required autocomplete="off"></div>
      <div class="field"><label for="ncBody">Message</label>
        <textarea id="ncBody" rows="6" maxlength="4000" required></textarea></div>
      <div class="case-actions">
        <button type="submit" class="primary" id="ncSubmit">File case and run triage</button>
        <button type="button" class="link-btn" id="ncExample">Use an example</button>
      </div>
      <div id="ncError"></div>
    </form>`;
  $("ncExample").addEventListener("click", fillExample);
  $("newCaseForm").addEventListener("submit", submitNewCase);
  $("ncSubject").focus();
}

function fillExample() {
  const list = examples();
  const ex = list[exampleIndex % list.length];
  exampleIndex += 1;
  $("ncSender").value = ex.sender;
  $("ncType").value = ex.sender_type;
  $("ncSubject").value = ex.subject;
  $("ncBody").value = ex.body;
}

async function submitNewCase(e) {
  e.preventDefault();
  const submit = $("ncSubmit");
  const errorBox = $("ncError");
  errorBox.innerHTML = "";
  submit.disabled = true;
  submit.textContent = "Filing case\u2026";

  const payload = {
    sender: $("ncSender").value.trim() || "Anonymous",
    sender_type: $("ncType").value,
    subject: $("ncSubject").value.trim(),
    body: $("ncBody").value.trim(),
  };

  try {
    const res = await fetch(`${API}/tickets`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
    });
    if (!res.ok) {
      errorBox.innerHTML = `<div class="error-box">${esc(await errorText(res, "Could not file the case."))}</div>`;
      submit.disabled = false;
      submit.textContent = "File case and run triage";
      return;
    }
    const { ticket, possible_duplicates } = await res.json();
    selectedId = ticket.id;
    await loadTickets();
    renderCase(ticket);
    renderDuplicates(possible_duplicates);
    await runClassify(ticket);
  } catch {
    errorBox.innerHTML = `<div class="error-box">Could not reach the backend.</div>`;
    submit.disabled = false;
    submit.textContent = "File case and run triage";
  }
}

// ------------------------------------------------------------------ ask
async function submitAsk(e) {
  e.preventDefault();
  const question = askInput.value.trim();
  if (!question) return;
  askAnswer.hidden = false;
  askAnswer.innerHTML = `<p class="ask-loading">Looking through the cases&hellip;</p>`;
  try {
    const res = await fetch(`${API}/ask`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question }),
    });
    if (!res.ok) {
      askAnswer.innerHTML = `<div class="error-box">${esc(await errorText(res, "Could not answer that."))}</div>`;
      return;
    }
    const data = await res.json();
    const refs = data.referenced_ticket_ids || [];
    askAnswer.innerHTML = `<p>${esc(data.answer)}</p>${refs.length ? `<p class="ask-sources">Based on: ${refs.map(esc).join(", ")}</p>` : ""}`;
  } catch {
    askAnswer.innerHTML = `<div class="error-box">Could not reach the backend.</div>`;
  }
}

init();
