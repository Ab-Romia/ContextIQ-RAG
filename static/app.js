"use strict";

const $ = (id) => document.getElementById(id);
const els = {
  docText: $("doc-text"),
  docFile: $("doc-file"),
  indexBtn: $("index-btn"),
  clearBtn: $("clear-btn"),
  kbStatus: $("kb-status"),
  apiKey: $("api-key"),
  model: $("model"),
  rerank: $("rerank-toggle"),
  query: $("query"),
  askBtn: $("ask-btn"),
  answer: $("answer"),
  traceCard: $("trace-card"),
  traceSummary: $("trace-summary"),
  traceBody: document.querySelector("#trace-table tbody"),
};

// Persist the key and model locally so a reload does not lose them. They are stored in
// this browser only; the server never receives them except as a per-request credential.
els.apiKey.value = localStorage.getItem("contextiq.key") || "";
els.model.value = localStorage.getItem("contextiq.model") || "";
els.apiKey.addEventListener("change", () => localStorage.setItem("contextiq.key", els.apiKey.value.trim()));
els.model.addEventListener("change", () => localStorage.setItem("contextiq.model", els.model.value.trim()));

function setStatus(node, message, kind) {
  node.textContent = message;
  node.className = "status" + (kind ? " " + kind : "");
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

// A deliberately small renderer: escape everything first, then re-introduce a handful of
// safe inline forms. No third-party markdown library, so no untrusted HTML reaches innerHTML.
function renderInline(text) {
  return escapeHtml(text)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/\[(\d+)\]/g, '<span class="cite" data-marker="$1" role="button" tabindex="0">[$1]</span>');
}

function renderAnswer(text) {
  const blocks = text.split(/\n\n+/);
  const html = blocks
    .map((block) => {
      const lines = block.split("\n");
      const isList = lines.every((l) => /^\s*[-*]\s+/.test(l));
      if (isList) {
        const items = lines.map((l) => "<li>" + renderInline(l.replace(/^\s*[-*]\s+/, "")) + "</li>").join("");
        return "<ul>" + items + "</ul>";
      }
      return "<p>" + renderInline(block) + "</p>";
    })
    .join("");
  els.answer.innerHTML = html;
  bindCitations();
}

function bindCitations() {
  els.answer.querySelectorAll(".cite").forEach((chip) => {
    const jump = () => {
      const row = els.traceBody.querySelector('tr[data-marker="' + chip.dataset.marker + '"]');
      if (row) {
        row.scrollIntoView({ behavior: "smooth", block: "center" });
        row.animate([{ background: "rgba(52,211,153,0.3)" }, { background: "transparent" }], { duration: 1200 });
      }
    };
    chip.onclick = jump;
    chip.onkeydown = (e) => { if (e.key === "Enter") jump(); };
  });
}

function fmt(value, digits) {
  return value === null || value === undefined ? "" : Number(value).toFixed(digits);
}

function renderTrace(trace) {
  els.traceCard.hidden = false;
  const mode = trace.rerank_enabled ? "hybrid + rerank" : "hybrid (no rerank)";
  els.traceSummary.textContent =
    `${mode} | dense ${trace.dense_count}, BM25 ${trace.sparse_count}, fused ${trace.fused_count}, ` +
    `showing top ${trace.candidates.length}, ${trace.citations.length} sent to the model`;

  els.traceBody.innerHTML = "";
  trace.candidates.forEach((c, i) => {
    const tr = document.createElement("tr");
    if (c.selected) tr.className = "selected";
    tr.dataset.marker = c.selected ? String(selectedMarker(trace, c)) : "";
    tr.innerHTML =
      `<td class="num">${i + 1}</td>` +
      `<td>${escapeHtml(c.source)} <span class="muted-cell">#${c.ordinal}</span></td>` +
      `<td class="num ${c.dense_rank === null ? "muted-cell" : ""}">${c.dense_rank === null ? "-" : c.dense_rank}</td>` +
      `<td class="num ${c.sparse_rank === null ? "muted-cell" : ""}">${c.sparse_rank === null ? "-" : c.sparse_rank}</td>` +
      `<td class="num">${fmt(c.rrf_score, 4)}</td>` +
      `<td class="num ${c.rerank_score === null ? "muted-cell" : ""}">${c.rerank_score === null ? "-" : fmt(c.rerank_score, 2)}</td>` +
      `<td class="preview">${escapeHtml(c.preview)}</td>`;
    els.traceBody.appendChild(tr);
  });
}

// Map a selected candidate back to its citation marker by matching the chunk text.
function selectedMarker(trace, candidate) {
  const preview = candidate.preview.replace(/\.\.\.$/, "");
  const cite = trace.citations.find((c) => c.text.startsWith(preview.slice(0, 60)));
  return cite ? cite.marker : "";
}

async function indexPayload(url, options, label) {
  setStatus(els.kbStatus, label, null);
  els.indexBtn.disabled = true;
  try {
    const res = await fetch(url, options);
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Indexing failed.");
    setStatus(els.kbStatus, `Indexed ${data.chunks_added} chunks from "${data.source}". Total in store: ${data.chunks_total}.`, "ok");
  } catch (err) {
    setStatus(els.kbStatus, err.message, "err");
  } finally {
    els.indexBtn.disabled = false;
  }
}

els.indexBtn.addEventListener("click", () => {
  const text = els.docText.value.trim();
  if (!text) { setStatus(els.kbStatus, "Paste some text or upload a file first.", "err"); return; }
  indexPayload("/api/index", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, source: "pasted-text" }),
  }, "Indexing pasted text...");
});

els.docFile.addEventListener("change", () => {
  const file = els.docFile.files[0];
  if (!file) return;
  const form = new FormData();
  form.append("file", file);
  indexPayload("/api/index-file", { method: "POST", body: form }, `Reading "${file.name}"...`);
  els.docFile.value = "";
});

els.clearBtn.addEventListener("click", async () => {
  await fetch("/api/clear", { method: "POST" });
  setStatus(els.kbStatus, "Knowledge base cleared.", null);
  els.traceCard.hidden = true;
  els.answer.innerHTML = "";
});

function parseSSE(buffer, onEvent) {
  const blocks = buffer.split("\n\n");
  const remainder = blocks.pop();
  for (const block of blocks) {
    let event = "message";
    let data = "";
    for (const line of block.split("\n")) {
      if (line.startsWith("event:")) event = line.slice(6).trim();
      else if (line.startsWith("data:")) data += line.slice(5).trim();
    }
    if (data) onEvent(event, data);
  }
  return remainder;
}

async function ask() {
  const query = els.query.value.trim();
  const apiKey = els.apiKey.value.trim();
  if (!query) return;
  if (!apiKey) { els.answer.innerHTML = '<p class="status err">Enter an API key to generate an answer. Retrieval still runs, but the model needs a key.</p>'; }

  els.askBtn.disabled = true;
  els.answer.innerHTML = '<p class="status">Retrieving...</p>';
  let answer = "";

  try {
    const res = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query,
        api_key: apiKey || "none",
        model: els.model.value.trim() || null,
        rerank: els.rerank.checked,
      }),
    });

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let started = false;

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      buffer = parseSSE(buffer, (event, data) => {
        const payload = JSON.parse(data);
        if (event === "trace") {
          renderTrace(payload);
          els.answer.innerHTML = '<p class="status">Generating...</p>';
        } else if (event === "token") {
          if (!started) { answer = ""; started = true; }
          answer += payload.text;
          renderAnswer(answer);
          const cursor = document.createElement("span");
          cursor.className = "cursor";
          els.answer.appendChild(cursor);
        } else if (event === "error") {
          els.answer.innerHTML = '<p class="status err">' + escapeHtml(payload.detail) + "</p>";
        }
      });
    }
    if (started) renderAnswer(answer);
  } catch (err) {
    els.answer.innerHTML = '<p class="status err">Request failed: ' + escapeHtml(err.message) + "</p>";
  } finally {
    els.askBtn.disabled = false;
  }
}

els.askBtn.addEventListener("click", ask);
els.query.addEventListener("keydown", (e) => { if (e.key === "Enter") ask(); });
