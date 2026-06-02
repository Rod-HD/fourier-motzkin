/* Fourier-Motzkin LP Solver v2 — logic giao diện */
"use strict";

let N = 2;
let sense = "max";
let method = "algebraic";
let mode = "form";
let chart = null;
let lastResult = null;

const SUB = "₀₁₂₃₄₅₆₇₈₉";
const sub = (k) => String(k).split("").map((d) => SUB[+d]).join("");
const vlab = (i) => "x" + sub(i + 1);
const esc = (s) => (s == null ? "" : String(s)
  .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;"));


/* ---- thiết lập cơ bản ---- */
function setSense(s) {
  sense = s;
  document.getElementById("tgMax").classList.toggle("on", s === "max");
  document.getElementById("tgMin").classList.toggle("on", s === "min");
}
function setMethod(m) {
  method = m;
  document.getElementById("mAlg").classList.toggle("on", m === "algebraic");
  document.getElementById("mGeo").classList.toggle("on", m === "geometric");
}
function setMode(m) {
  mode = m;
  document.getElementById("tgForm").classList.toggle("on", m === "form");
  document.getElementById("tgText").classList.toggle("on", m === "text");
  document.getElementById("textBox").classList.toggle("show", m === "text");
  document.getElementById("cList").style.display = m === "form" ? "" : "none";
  document.getElementById("addC").style.display = m === "form" ? "block" : "none";
  document.getElementById("nnWrap").style.display = m === "form" ? "flex" : "none";
}
function applyN() {
  const v = parseInt(document.getElementById("nIn").value, 10);
  if (isNaN(v) || v < 1) { alert("Số biến phải ≥ 1"); document.getElementById("nIn").value = N; return; }
  renderObjective(v);
}

/* ---- hàm mục tiêu ---- */
function renderObjective(n) {
  N = n;
  const line = document.getElementById("objLine");
  const old = [...line.querySelectorAll(".oc")].map((i) => i.value);
  line.innerHTML = "";
  for (let i = 0; i < n; i++) {
    const term = document.createElement("span");
    term.className = "obj-term";
    if (i > 0) {
      const plus = document.createElement("span");
      plus.style.color = "var(--slate)"; plus.textContent = "+";
      term.appendChild(plus);
    }
    const inp = document.createElement("input");
    inp.type = "text"; inp.className = "coef oc";
    inp.value = old[i] !== undefined ? old[i] : "1";
    inp.placeholder = "vd: 1/2";
    inp.addEventListener("input", previewObj);
    term.appendChild(inp);
    const lb = document.createElement("span");
    lb.className = "vlab"; lb.textContent = vlab(i);
    term.appendChild(lb);
    line.appendChild(term);
  }
  document.querySelectorAll(".crow").forEach((row) => {
    const oldc = [...row.querySelectorAll(".terms .coef")].map((i) => i.value);
    const sec = row.querySelector(".terms");
    sec.innerHTML = "";
    for (let i = 0; i < n; i++) sec.appendChild(coefCell(i, oldc[i] !== undefined ? oldc[i] : 0));
  });
  previewObj();
}
function coefCell(i, val) {
  const g = document.createElement("span");
  g.className = "obj-term";
  const inp = document.createElement("input");
  inp.type = "number"; inp.step = "any"; inp.className = "coef";
  inp.value = val;
  const lb = document.createElement("span");
  lb.className = "vlab"; lb.textContent = vlab(i);
  g.appendChild(inp); g.appendChild(lb);
  return g;
}
function previewObj() {
  const t = [];
  document.querySelectorAll(".oc").forEach((inp, i) => {
    const raw = inp.value.trim();
    if (!raw) return;
    const name = vlab(i);
    const num = evalFrac(raw);
    if (num === null) { t.push(`${raw}·${name}`); return; }
    if (num === 0) return;
    const disp = raw.includes("/") ? raw : (num === Math.trunc(num) ? String(num) : raw);
    if (!t.length) {
      t.push(num === 1 ? name : num === -1 ? "−" + name : disp + name);
    } else if (num < 0) {
      t.push("− " + (num === -1 ? name : (raw.startsWith("-") ? raw.slice(1) : raw) + name));
    } else {
      t.push("+ " + (num === 1 ? name : disp + name));
    }
  });
  document.getElementById("objPrev").textContent = t.length ? "z = " + t.join(" ") : "z = 0";
}

function evalFrac(s) {
  const m = s.match(/^\s*(-?\d+)\s*\/\s*(-?\d+)\s*$/);
  if (m) { const d = parseInt(m[2]); return d ? parseInt(m[1]) / d : null; }
  const n = parseFloat(s);
  return isNaN(n) ? null : n;
}

/* ---- ràng buộc form ---- */
function addRow(coeffs, s, rhs) {
  const row = document.createElement("div");
  row.className = "crow";
  const terms = document.createElement("span");
  terms.className = "terms";
  for (let i = 0; i < N; i++) terms.appendChild(coefCell(i, Array.isArray(coeffs) ? (coeffs[i] ?? 0) : 0));
  row.appendChild(terms);
  const sel = document.createElement("select");
  sel.className = "sense";
  [["<=", "≤"], [">=", "≥"], ["=", "="]].forEach(([v, txt]) => {
    const o = document.createElement("option");
    o.value = v; o.textContent = txt;
    if (v === (s || "<=")) o.selected = true;
    sel.appendChild(o);
  });
  row.appendChild(sel);
  const ri = document.createElement("input");
  ri.type = "number"; ri.step = "any"; ri.className = "rhs"; ri.value = rhs !== undefined ? rhs : 0;
  row.appendChild(ri);
  const del = document.createElement("button");
  del.className = "del"; del.innerHTML = "&times;"; del.title = "Xoá";
  del.onclick = () => row.remove();
  row.appendChild(del);
  document.getElementById("cList").appendChild(row);
}

function loadExample() {
  N = 2;
  document.getElementById("nIn").value = 2;
  setMode("form"); setSense("max"); setMethod("algebraic");
  renderObjective(2);
  const oc = document.querySelectorAll(".oc");
  oc[0].value = "3"; oc[1].value = "2";
  document.getElementById("cList").innerHTML = "";
  addRow([1, 1], "<=", 4);
  addRow([2, 1], "<=", 6);
  document.getElementById("nnChk").checked = true;
  document.getElementById("cText").value = ["x1 + x2 <= 4", "2x1 + x2 <= 6", "-x1 <= 0", "-x2 <= 0"].join("\n");
  previewObj();
}
function resetAll() {
  N = 2;
  document.getElementById("nIn").value = 2;
  setMode("form"); setSense("max"); setMethod("algebraic");
  renderObjective(2);
  document.getElementById("cList").innerHTML = "";
  document.getElementById("cText").value = "";
  document.getElementById("nnChk").checked = true;
  previewObj();
  document.getElementById("resultSec").style.display = "none";
  destroyChart();
  lastResult = null;
}

function destroyChart() {
  if (chart) {
    if (chart._raf) cancelAnimationFrame(chart._raf);
    chart.destroy();
    chart = null;
  }
  document.querySelectorAll(".pulse-dot").forEach((e) => e.remove());
}

/* ---- thu thập input ---- */
function collect() {
  const obj = [...document.querySelectorAll(".oc")].map((i) => i.value.trim() || "0");
  if (mode === "text") {
    return { n: N, obj, sense, method, input_mode: "text", constraints_text: document.getElementById("cText").value };
  }
  const constraints = [];
  document.querySelectorAll(".crow").forEach((row) => {
    const coeffs = [...row.querySelectorAll(".terms .coef")].map((i) => parseFloat(i.value) || 0);
    constraints.push({ coeffs, sense: row.querySelector(".sense").value, rhs: parseFloat(row.querySelector(".rhs").value) || 0 });
  });
  if (document.getElementById("nnChk").checked) {
    for (let i = 0; i < N; i++) {
      const c = Array(N).fill(0); c[i] = -1;
      constraints.push({ coeffs: c, sense: "<=", rhs: 0 });
    }
  }
  return { n: N, obj, sense, method, input_mode: "form", constraints };
}

/* ---- gọi API giải ---- */
async function runSolve() {
  const btn = document.getElementById("solve");
  document.getElementById("solveTxt").style.display = "none";
  document.getElementById("solveSpin").style.display = "inline-block";
  btn.disabled = true;
  try {
    const res = await fetch("/api/solve", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(collect()),
    });
    const data = await res.json();
    if (!res.ok) { showError(data); return; }
    lastResult = data;
    render(data);
  } catch (e) {
    showError({ error: String(e) });
  } finally {
    document.getElementById("solveTxt").style.display = "inline";
    document.getElementById("solveSpin").style.display = "none";
    btn.disabled = false;
  }
}

function showError(d) {
  document.getElementById("resultSec").style.display = "block";
  const details = Array.isArray(d.details) ? d.details : [];
  const dh = details.length ? `<ul style="margin:6px 0 0 18px">${details.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>` : "";
  document.getElementById("resultCard").innerHTML =
    `<div class="panel-b"><div class="alert err"><span class="ic">⚠️</span>
     <div><strong>Không xử lý được</strong><div>${esc(d.error || "Lỗi không xác định")}</div>${dh}</div></div></div>`;
  ["steps", "chart", "trace", "export"].forEach((t) => { document.getElementById("pane-" + t).innerHTML = ""; });
}

/* ---- hiển thị kết quả ---- */
function render(d) {
  document.getElementById("resultSec").style.display = "block";
  renderResultCard(d);
  renderSteps(d);
  renderTrace(d);
  renderExport(d);
  renderChart(d);
  switchTab("steps");
}

function renderResultCard(d) {
  const card = document.getElementById("resultCard");
  if (d.status === "feasible") {
    let chips = "";
    Object.entries(d.solution).forEach(([k, v]) => {
      chips += `<div class="chip"><span class="k">${k}</span><span class="v">${esc(v.fraction)}</span><span class="d">≈ ${esc(v.decimal)}</span></div>`;
    });
    chips += `<div class="chip z"><span class="k">z*</span><span class="v">${esc(d.z.fraction)}</span><span class="d">≈ ${esc(d.z.decimal)}</span></div>`;
    card.innerHTML = `<div class="panel-b"><div class="alert ok"><span class="ic">✓</span>
      <div><strong>Bài toán có nghiệm tối ưu</strong> — phương pháp ${d.method === "geometric" ? "hình học" : "đại số"}</div></div>
      <div class="chips" style="margin-top:11px">${chips}</div></div>`;
  } else {
    const msg = d.status === "unbounded" ? "Bài toán KHÔNG BỊ CHẶN — hàm mục tiêu tăng vô hạn." : "Bài toán VÔ NGHIỆM — miền khả thi rỗng.";
    card.innerHTML = `<div class="panel-b"><div class="alert err"><span class="ic">⚠️</span><div><strong>${esc(msg)}</strong></div></div></div>`;
  }
}

function renderSteps(d) {
  const host = document.getElementById("pane-steps");
  const inp = d.input || {};
  let html = `<div class="step open"><div class="step-h" onclick="toggleStep(this)">
      <span class="step-n">IN</span> Bài toán đầu vào <span style="margin-left:auto">▾</span></div>
      <div class="step-b">
        <div class="row-eq" style="background:var(--teal-l)">${esc(inp.sense || "").toUpperCase()} z = ${esc(inp.objective || "")}</div>
        ${(inp.constraints || []).map((c) => `<div class="row-eq">${esc(c)}</div>`).join("")}
      </div></div>`;
  if (Array.isArray(d.steps) && d.steps.length) {
    d.steps.forEach((st) => {
      html += `<div class="step"><div class="step-h" onclick="toggleStep(this)">
        <span class="step-n">${st.step}</span> ${esc(st.title)} <span style="margin-left:auto">▾</span></div>
        <div class="step-b">${st.rows.map((r) => `<div class="row-eq">${esc(r)}</div>`).join("")}</div></div>`;
    });
  } else if (d.method === "geometric") {
    html += `<div class="hint" style="margin-top:10px">Phương pháp hình học không dùng các bước khử đại số; xem tab “Diễn giải” và “Biểu đồ”.</div>`;
  }
  host.innerHTML = html;
}
function toggleStep(h) { h.parentElement.classList.toggle("open"); }

function renderTrace(d) {
  const items = (d.trace && d.trace.items) || [];
  const host = document.getElementById("pane-trace");
  host.innerHTML = items.map((it) => {
    const body = it.body ? `<div class="tr-body">${esc(it.body)}</div>` : "";
    const formula = it.formula ? `<div class="tr-formula">${esc(it.formula)}</div>` : "";
    const out = it.outcome ? `<div class="tr-out">⇒ ${esc(it.outcome)}</div>` : "";
    return `<div class="tr-item ${it.kind}"><div class="tr-h">
      <span class="tr-title">${esc(it.title)}</span></div>${body}${formula}${out}</div>`;
  }).join("");
}

function renderExport(d) {
  const host = document.getElementById("pane-export");
  host.innerHTML = `<div class="exp-bar">
      <button class="btn btn-main btn-sm" onclick="downloadTxt()">⬇ Tải file .txt</button>
      <button class="btn btn-ghost btn-sm" onclick="copyJson()">Sao chép JSON</button>
    </div><div class="exp-pre" id="expPre">${esc(buildPreview(d))}</div>`;
}

function buildPreview(d) {
  const inp = d.input || {};
  const L = [];
  L.push(`${(inp.sense || "").toUpperCase()} z = ${inp.objective || ""}`);
  L.push("Ràng buộc (đã chuẩn hóa ≤):");
  (inp.constraints || []).forEach((c) => L.push("  " + c));
  L.push("");
  if (d.status === "feasible") {
    Object.entries(d.solution).forEach(([k, v]) => L.push(`${k} = ${v.fraction}  (≈ ${v.decimal})`));
    L.push(`z* = ${d.z.fraction}  (≈ ${d.z.decimal})`);
  } else {
    L.push(d.status === "unbounded" ? "KHÔNG BỊ CHẶN" : "VÔ NGHIỆM");
  }
  return L.join("\n");
}

async function downloadTxt() {
  if (!lastResult) return;
  const res = await fetch("/api/export", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ result: lastResult }),
  });
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = "giai_trinh_v2.txt"; a.click();
  URL.revokeObjectURL(url);
}
function copyJson() {
  navigator.clipboard.writeText(JSON.stringify(lastResult, null, 2)).then(showToast);
}
function showToast() {
  const t = document.getElementById("toast");
  t.classList.add("show");
  setTimeout(() => t.classList.remove("show"), 1500);
}

function switchTab(t) {
  document.querySelectorAll(".tab").forEach((b) => b.classList.toggle("on", b.dataset.t === t));
  ["steps", "chart", "trace", "export"].forEach((p) => document.getElementById("pane-" + p).classList.toggle("on", p === t));
}

/* ---- biểu đồ (n=2) ---- */
function renderChart(d) {
  const host = document.getElementById("pane-chart");
  const chartData = d.chart;
  if (!chartData || (d.input && d.input.n !== 2)) {
    host.innerHTML = `<div class="chart-note">Biểu đồ chỉ hiển thị cho bài toán 2 biến.</div>`;
    return;
  }
  const verts = chartData.vertices || [];
  if (d.status === "infeasible") {
    // Vô nghiệm: vẫn vẽ các đường ràng buộc để thấy KHÔNG có vùng giao chung.
    host.innerHTML =
      `<div class="chart-note" style="margin-bottom:12px">Bài toán <strong>VÔ NGHIỆM</strong>: các nửa mặt phẳng ràng buộc
       không có vùng giao chung, nên không tồn tại miền khả thi (đa giác rỗng).
       Biểu đồ dưới chỉ vẽ các đường ràng buộc để thấy rõ điều đó.</div>
       <div id="chartHost"><canvas id="cv" height="420"></canvas></div>`;
    drawChart(chartData, d);
    return;
  }
  if (d.status === "unbounded") {
    host.innerHTML = `<div class="chart-note">Bài toán <strong>KHÔNG BỊ CHẶN</strong>: miền khả thi mở vô hạn theo
      hướng tối ưu nên không có đỉnh tối ưu hữu hạn để vẽ.</div>`;
    return;
  }
  host.innerHTML = `<div id="chartHost"><canvas id="cv" height="420"></canvas></div>${vertexTable(chartData)}`;
  drawChart(chartData, d);
}

function vertexTable(chartData) {
  const vs = chartData.vertices || [];
  if (!vs.length) return "";
  let rows = vs.map((v) =>
    `<tr class="${v.is_optimal ? "opt" : ""}"><td>${esc(v.label)}</td>
      <td>(${esc(v.x_frac)}, ${esc(v.y_frac)})</td><td>${esc(v.z_frac)}</td>
      <td>${v.is_optimal ? "★ tối ưu" : ""}</td></tr>`).join("");
  return `<table class="vtable"><thead><tr><th>Đỉnh</th><th>Toạ độ</th><th>z</th><th></th></tr></thead><tbody>${rows}</tbody></table>`;
}

const CLRS = ["#e53935", "#1e88e5", "#43a047", "#fb8c00", "#8e24aa", "#00acc1", "#f4511e", "#6d4c41", "#c0ca33", "#00838f"];

function fmtNum(v) { return Number.isInteger(v) ? v : +(+v).toFixed(4); }

function drawChart(chartData, d) {
  destroyChart();
  const verts = chartData.vertices || [];
  const lines = chartData.boundary_lines || [];
  const level = chartData.level_lines || {};
  const sense = (level && level.sense) || (d.input && d.input.sense) || "max";

  // điểm khả thi và phạm vi trục
  const feasPts = verts.map((v) => v.point);
  // gộp thêm các giao điểm trục của đường ràng buộc để chọn khung hợp lý
  const xsCand = feasPts.map((p) => p[0]).concat([0]);
  const ysCand = feasPts.map((p) => p[1]).concat([0]);
  lines.forEach((ln) => {
    if (Math.abs(ln.a) > 1e-9) xsCand.push(ln.rhs / ln.a);
    if (Math.abs(ln.b) > 1e-9) ysCand.push(ln.rhs / ln.b);
  });
  const posX = xsCand.filter((v) => isFinite(v) && v >= 0);
  const posY = ysCand.filter((v) => isFinite(v) && v >= 0);
  const xMax = Math.max(...posX, 1) * 1.3;
  const yMax = Math.max(...posY, 1) * 1.3;
  const cx = feasPts.length ? feasPts.reduce((s, p) => s + p[0], 0) / feasPts.length : xMax / 2;
  const cy = feasPts.length ? feasPts.reduce((s, p) => s + p[1], 0) / feasPts.length : yMax / 2;

  const optV = verts.find((v) => v.is_optimal) || null;
  const feasNO = verts.filter((v) => !v.is_optimal);

  const datasets = [];

  // dataset giả để legend hiển thị "Miền nghiệm"
  datasets.push({
    type: "line", label: "Miền nghiệm", data: [],
    backgroundColor: "rgba(15,118,110,0.16)", borderColor: "rgba(15,118,110,0.55)",
    borderWidth: 2, pointRadius: 0,
  });

  // đường ràng buộc — mỗi đường một màu, có nhãn
  lines.forEach((ln, i) => {
    if (Math.abs(ln.a) < 1e-12 && Math.abs(ln.b) < 1e-12) return;
    const pts = lineClip(ln.a, ln.b, ln.rhs, xMax, yMax);
    if (pts.length < 2) return;
    datasets.push({
      type: "line", label: ln.label || ("d" + ln.idx), data: pts,
      borderColor: CLRS[i % CLRS.length], borderWidth: 1.5, borderDash: [5, 4],
      pointRadius: 0, fill: false, tension: 0, order: 9,
    });
  });

  // đường mức tối ưu z* (cam đậm, nét LIỀN, tĩnh)
  if (level && level.coeffs && optV && level.z_opt != null) {
    const pZS = lineClip(level.coeffs[0], level.coeffs[1], level.z_opt, xMax, yMax);
    if (pZS.length >= 2) datasets.push({
      type: "line", label: `z* = ${fmtNum(level.z_opt)} (tối ưu)`, data: pZS,
      borderColor: "#FF6F00", borderWidth: 2.5, borderDash: [],
      pointRadius: 0, fill: false, tension: 0, order: 7,
    });
  }

  // đỉnh khả thi (không tối ưu)
  if (feasNO.length) datasets.push({
    type: "scatter", label: "Đỉnh khả thi",
    data: feasNO.map((v) => ({ x: v.point[0], y: v.point[1], z: v.z_frac, lbl: v.label })),
    backgroundColor: "#0f766e", borderColor: "#0f766e",
    pointRadius: 6, pointHoverRadius: 8, order: 2,
  });

  // đỉnh tối ưu (nổi bật, viền vàng)
  if (optV) datasets.push({
    type: "scatter", label: `★ Tối ưu (z=${optV.z_frac})`,
    data: [{ x: optV.point[0], y: optV.point[1], z: optV.z_frac, lbl: "★ " + optV.label }],
    backgroundColor: "#0f766e", borderColor: "#FFD600",
    borderWidth: 3, pointRadius: 9, pointHoverRadius: 11, order: 1,
  });

  // ── Plugin 1: tô miền nghiệm ──────────────────────────────────────────
  // feasPts >= 3 → đa giác có diện tích; = 2 → miền suy biến thành ĐOẠN
  // (vd ràng buộc đẳng thức x1+x2=4); = 1 → miền là MỘT điểm.
  const feasibleRegionPlugin = {
    id: "feasibleRegion",
    beforeDatasetsDraw(ch) {
      if (feasPts.length < 1) return;
      const xs = ch.scales.x, ys = ch.scales.y;
      const px = (p) => xs.getPixelForValue(p[0]);
      const py = (p) => ys.getPixelForValue(p[1]);
      const g = ch.ctx;
      g.save();
      if (feasPts.length === 1) {
        // miền suy biến thành một điểm: vẽ chấm tròn nhấn
        g.beginPath();
        g.arc(px(feasPts[0]), py(feasPts[0]), 6, 0, 2 * Math.PI);
        g.fillStyle = "rgba(15,118,110,0.16)";
        g.fill();
        g.strokeStyle = "rgba(15,118,110,0.9)";
        g.lineWidth = 2.5; g.setLineDash([]); g.stroke();
      } else if (feasPts.length === 2) {
        // miền suy biến thành đoạn thẳng: vẽ đoạn đậm nối 2 đỉnh
        g.beginPath();
        g.moveTo(px(feasPts[0]), py(feasPts[0]));
        g.lineTo(px(feasPts[1]), py(feasPts[1]));
        g.strokeStyle = "rgba(15,118,110,0.85)";
        g.lineWidth = 6; g.lineCap = "round"; g.setLineDash([]); g.stroke();
      } else {
        const sorted = [...feasPts].sort((a, b) =>
          Math.atan2(a[1] - cy, a[0] - cx) - Math.atan2(b[1] - cy, b[0] - cx));
        g.beginPath();
        g.moveTo(px(sorted[0]), py(sorted[0]));
        for (let i = 1; i < sorted.length; i++) g.lineTo(px(sorted[i]), py(sorted[i]));
        g.closePath();
        g.fillStyle = "rgba(15,118,110,0.16)";
        g.fill();
        g.strokeStyle = "rgba(15,118,110,0.55)";
        g.lineWidth = 2; g.setLineDash([]); g.stroke();
      }
      g.restore();
    },
  };

  // ── Plugin 2: mũi tên hướng miền + nhãn đỉnh ──────────────────────────
  const extrasPlugin = {
    id: "extras",
    afterDraw(ch) {
      const g = ch.ctx, xs = ch.scales.x, ys = ch.scales.y;
      const toPx = (wx, wy) => ({ px: xs.getPixelForValue(wx), py: ys.getPixelForValue(wy) });
      const arrow = (x1, y1, x2, y2, color, lw) => {
        const dx = x2 - x1, dy = y2 - y1, len = Math.sqrt(dx * dx + dy * dy);
        if (len < 2) return;
        const ux = dx / len, uy = dy / len, hs = 9;
        g.save(); g.strokeStyle = color; g.fillStyle = color; g.lineWidth = lw;
        g.beginPath(); g.moveTo(x1, y1); g.lineTo(x2, y2); g.stroke();
        g.beginPath(); g.moveTo(x2, y2);
        g.lineTo(x2 - hs * ux + hs * 0.4 * uy, y2 - hs * uy - hs * 0.4 * ux);
        g.lineTo(x2 - hs * ux - hs * 0.4 * uy, y2 - hs * uy + hs * 0.4 * ux);
        g.closePath(); g.fill(); g.restore();
      };
      // mũi tên chỉ về phía miền khả thi cho từng ràng buộc
      lines.forEach((ln, i) => {
        if (Math.abs(ln.a) < 1e-12 && Math.abs(ln.b) < 1e-12) return;
        const pts = lineClip(ln.a, ln.b, ln.rhs, xMax, yMax);
        if (pts.length < 2) return;
        const mx = (pts[0].x + pts[1].x) / 2, my = (pts[0].y + pts[1].y) / 2;
        let nx = 0, ny = 0;
        if (ln.feasible_dir) { nx = ln.feasible_dir[0]; ny = ln.feasible_dir[1]; }
        const aLen = Math.max(xMax, yMax) * 0.06;
        const s = toPx(mx, my), e = toPx(mx + nx * aLen, my + ny * aLen);
        arrow(s.px, s.py, e.px, e.py, CLRS[i % CLRS.length], 1.5);
      });
      // nhãn đỉnh
      g.save(); g.font = "bold 11px Segoe UI, sans-serif";
      verts.forEach((v) => {
        const { px, py } = toPx(v.point[0], v.point[1]);
        const txt = (v.is_optimal ? "★ " : "") + v.label + ` (${v.x_frac}, ${v.y_frac})`;
        g.fillStyle = v.is_optimal ? "#b45309" : "#0b574f";
        g.textAlign = "left"; g.textBaseline = "bottom";
        g.fillText(txt, px + 7, py - 5);
      });
      g.restore();
    },
  };

  // ── Plugin 3: đường mục tiêu trượt từ xa về tối ưu ────────────────────
  const slidingObjective = {
    id: "slidingObjective",
    afterDatasetsDraw(ch) {
      const ld = ch._level;
      if (!ld || ld.z_opt == null || !ld.coeffs) return;
      const [c1, c2] = ld.coeffs;
      if (Math.abs(c1) < 1e-12 && Math.abs(c2) < 1e-12) return;
      const xA = ch.scales.x, yA = ch.scales.y;
      const sgn = sense === "max" ? -1 : 1;
      const xFar = sgn * c1 >= 0 ? xA.max : xA.min;
      const yFar = sgn * c2 >= 0 ? yA.max : yA.min;
      const zInit = c1 * xFar + c2 * yFar;
      const zOpt = ld.z_opt;

      const DUR = 4200;
      const raw = (performance.now() % DUR) / DUR;
      const t = raw < 0.5 ? 2 * raw * raw : 1 - Math.pow(-2 * raw + 2, 2) / 2;
      const zCur = zInit + (zOpt - zInit) * t;
      const op = 0.25 + 0.65 * t;

      const g = ch.ctx;
      const seg = (z) => (Math.abs(c2) > 1e-9
        ? [{ x: xA.min, y: (z - c1 * xA.min) / c2 }, { x: xA.max, y: (z - c1 * xA.max) / c2 }]
        : [{ x: z / c1, y: yA.min }, { x: z / c1, y: yA.max }]);
      const p = seg(zCur);
      const x1 = xA.getPixelForValue(p[0].x), y1 = yA.getPixelForValue(p[0].y);
      const x2 = xA.getPixelForValue(p[1].x), y2 = yA.getPixelForValue(p[1].y);

      g.save();
      g.beginPath(); g.moveTo(x1, y1); g.lineTo(x2, y2);
      g.strokeStyle = `rgba(255,111,0,${op})`; g.lineWidth = 2.2; g.setLineDash([8, 4]); g.stroke();

      const xm = (xA.min + xA.max) / 2;
      const ym = Math.abs(c2) > 1e-9 ? (zCur - c1 * xm) / c2 : (yA.min + yA.max) / 2;
      const pxm = xA.getPixelForValue(xm), pym = yA.getPixelForValue(ym);
      g.setLineDash([]);
      g.fillStyle = `rgba(230,81,0,${op})`; g.font = "11px Segoe UI, sans-serif";
      g.fillText("z = " + fmtNum(zCur), pxm + 6, pym - 6);

      const dx = (ld.direction && ld.direction[0]) || 0;
      const dy = (ld.direction && ld.direction[1]) || 0;
      const ax = pxm + dx * 40, ay = pym - dy * 40;
      g.beginPath(); g.moveTo(pxm, pym); g.lineTo(ax, ay);
      g.strokeStyle = `rgba(255,111,0,${op})`; g.lineWidth = 2; g.stroke();
      const ang = Math.atan2(ay - pym, ax - pxm), hl = 9;
      g.beginPath(); g.moveTo(ax, ay);
      g.lineTo(ax - hl * Math.cos(ang - Math.PI / 6), ay - hl * Math.sin(ang - Math.PI / 6));
      g.lineTo(ax - hl * Math.cos(ang + Math.PI / 6), ay - hl * Math.sin(ang + Math.PI / 6));
      g.closePath(); g.fillStyle = `rgba(255,111,0,${op})`; g.fill();
      g.fillText(sense === "max" ? "→ tăng z" : "→ giảm z", ax + 4, ay - 4);
      g.restore();

      ch._raf = requestAnimationFrame(() => { if (ch.canvas) ch.draw(); });
    },
  };

  const ctx = document.getElementById("cv");
  chart = new Chart(ctx, {
    type: "scatter",
    data: { datasets },
    plugins: [feasibleRegionPlugin, slidingObjective, extrasPlugin],
    options: {
      responsive: true, maintainAspectRatio: false, animation: { duration: 350 },
      scales: {
        x: { min: 0, max: Math.ceil(xMax), title: { display: true, text: "x₁" }, grid: { color: "#eef1f5" } },
        y: { min: 0, max: Math.ceil(yMax), title: { display: true, text: "x₂" }, grid: { color: "#eef1f5" } },
      },
      plugins: {
        legend: { position: "top", labels: { font: { size: 11 }, boxWidth: 16 } },
        tooltip: {
          callbacks: {
            label: (c) => {
              const p = c.raw;
              if (!p || p.x == null) return "";
              let s = p.lbl ? `${p.lbl} (${fmtNum(p.x)}, ${fmtNum(p.y)})` : `(${fmtNum(p.x)}, ${fmtNum(p.y)})`;
              if (p.z != null) s += `  →  z = ${p.z}`;
              return s;
            },
          },
        },
      },
    },
  });

  if (level && level.coeffs && level.z_opt != null) chart._level = level;

  const hostEl = document.getElementById("chartHost");
  if (optV && hostEl) {
    setTimeout(() => {
      if (!chart || !chart.scales) return;
      const px = chart.scales.x.getPixelForValue(optV.point[0]);
      const py = chart.scales.y.getPixelForValue(optV.point[1]);
      const m = document.createElement("div");
      m.className = "pulse-dot";
      m.style.left = px + "px"; m.style.top = py + "px";
      hostEl.appendChild(m);
    }, 420);
  }
}

// Cắt đường a1*x + a2*y = b vào khung [0,xMax] x [0,yMax].
function lineClip(a1, a2, b, xMax, yMax) {
  const pts = [];
  const add = (x, y) => {
    if (x >= -0.01 && x <= xMax + 0.01 && y >= -0.01 && y <= yMax + 0.01) pts.push({ x, y });
  };
  if (Math.abs(a2) > 1e-10) {
    add(0, b / a2); add(xMax, (b - a1 * xMax) / a2);
    if (Math.abs(a1) > 1e-10) { add(b / a1, 0); add((b - a2 * yMax) / a1, yMax); }
  } else if (Math.abs(a1) > 1e-10) {
    const x = b / a1; add(x, 0); add(x, yMax);
  }
  // khử trùng lặp, giữ tối đa 2 điểm
  const uniq = [];
  pts.forEach((p) => {
    if (!uniq.some((q) => Math.abs(q.x - p.x) < 1e-6 && Math.abs(q.y - p.y) < 1e-6)) uniq.push(p);
  });
  return uniq.slice(0, 2);
}

/* ════════════════════════════════════════════════════════════════
   KIỂM TRA PHỤ THUỘC VÒNG LẶP — UI tổng quát
════════════════════════════════════════════════════════════════ */

// Trạng thái hiện tại của form phụ thuộc
let depD = 1, depM = 1;

// Xây dựng form động theo d vòng lặp và m chiều mảng
function depBuild(d, m) {
  depD = d !== undefined ? d : parseInt(document.getElementById("dep_d").value, 10) || 1;
  depM = m !== undefined ? m : parseInt(document.getElementById("dep_m").value, 10) || 1;
  depD = Math.max(1, Math.min(4, depD));
  depM = Math.max(1, Math.min(4, depM));
  document.getElementById("dep_d").value = depD;
  document.getElementById("dep_m").value = depM;

  const host = document.getElementById("depForm");
  // Dùng i1, i2, ... (ASCII) thay vì i₁, i₂ để tránh lỗi font trong monospace
  const iLabels = Array.from({length: depD}, (_, j) => `i${j+1}`);

  // ── Phần 1: Cận vòng lặp ──────────────────────────────────────
  let html = `<div class="dep-section">
    <div class="dep-section-title">Cận vòng lặp (L ≤ i_k ≤ U)</div>`;
  for (let j = 0; j < depD; j++) {
    html += `<div class="dep-loop-row">
      <span class="lab">Vòng ${j+1} (${iLabels[j]})</span>
      <span class="lab">L</span>
      <input type="number" id="dep_L${j}" class="num-sm" value="0">
      <span class="lab">U</span>
      <input type="number" id="dep_U${j}" class="num-sm" value="10">
    </div>`;
  }
  html += `</div>`;

  // ── Phần 2: Hàm chỉ số GHI ────────────────────────────────────
  html += `<div class="dep-section">
    <div class="dep-section-title">
      Hàm chỉ số GHI
      <span style="font-weight:400;color:var(--slate);font-size:.78rem;margin-left:6px">
        — ô mảng được <strong style="color:#b45309">ghi</strong> ở mỗi lần lặp.
        Mỗi hàng k là một chiều của mảng. Nhập hệ số nhân với từng biến đếm i và hằng số cộng thêm.
      </span>
    </div>
    ${depAccessTable("w", iLabels)}
  </div>`;

  // ── Phần 3: Hàm chỉ số ĐỌC ────────────────────────────────────
  html += `<div class="dep-section">
    <div class="dep-section-title">
      Hàm chỉ số ĐỌC
      <span style="font-weight:400;color:var(--slate);font-size:.78rem;margin-left:6px">
        — ô mảng được <strong style="color:#15803d">đọc</strong> ở mỗi lần lặp.
        Cùng định dạng với hàm chỉ số ghi.
      </span>
    </div>
    ${depAccessTable("r", iLabels)}
  </div>`;

  // ── Phần 4: Preview vòng lặp đang mô phỏng ──────────────────
  html += `<div class="dep-section">
    <div class="dep-section-title">Vòng lặp đang mô phỏng</div>
    <div id="depEqPreview" class="dep-preview-box"></div>
  </div>`;

  host.innerHTML = html;

  // Gắn sự kiện cập nhật preview
  host.querySelectorAll("input").forEach(inp => inp.addEventListener("input", depUpdatePreview));
  depUpdatePreview();
  document.getElementById("depResult").innerHTML = "";
}

function depAccessTable(rw, iLabels) {
  const d = depD, m = depM;
  let th = `<th title="Chiều k của mảng (k=1 là chiều đầu tiên)">Chiều k</th>`;
  for (let j = 0; j < d; j++) {
    th += `<th title="Hệ số nhân với ${iLabels[j]} (biến đếm vòng ${j+1})">× ${iLabels[j]}</th>`;
  }
  th += `<th class="const-col" title="Hằng số cộng thêm (độ dịch cố định)">+ hằng số</th>`;
  let rows = "";
  for (let k = 0; k < m; k++) {
    let tds = `<td style="font-weight:700;color:var(--teal-d);text-align:center">k=${k+1}</td>`;
    for (let j = 0; j < d; j++) {
      const def = (j === 0 && k === 0) ? 1 : 0;
      tds += `<td><input type="number" id="dep_${rw}_${k}_${j}" value="${def}" placeholder="0"></td>`;
    }
    const defC = (rw === "r" && k === 0) ? -1 : 0;
    tds += `<td class="const-col"><input type="number" id="dep_${rw}_${k}_c" value="${defC}" placeholder="0"></td>`;
    rows += `<tr>${tds}</tr>`;
  }
  return `<table class="dep-matrix"><thead><tr>${th}</tr></thead><tbody>${rows}</tbody></table>`;
}

function depUpdatePreview() {
  const d = depD, m = depM;
  // Dùng i1, i2, ... (ASCII) để tránh lỗi font monospace với Unicode subscript
  const iLabels = Array.from({length: d}, (_, j) => `i${j+1}`);

  // Thu thập giá trị
  const loops = [];
  for (let j = 0; j < d; j++) {
    loops.push({
      L: parseInt(document.getElementById(`dep_L${j}`)?.value, 10) || 0,
      U: parseInt(document.getElementById(`dep_U${j}`)?.value, 10) || 0,
    });
  }
  const wCoeffs = [], rCoeffs = [], wConsts = [], rConsts = [];
  for (let k = 0; k < m; k++) {
    const wRow = [], rRow = [];
    for (let j = 0; j < d; j++) {
      wRow.push(parseInt(document.getElementById(`dep_w_${k}_${j}`)?.value, 10) || 0);
      rRow.push(parseInt(document.getElementById(`dep_r_${k}_${j}`)?.value, 10) || 0);
    }
    wCoeffs.push(wRow); rCoeffs.push(rRow);
    wConsts.push(parseInt(document.getElementById(`dep_w_${k}_c`)?.value, 10) || 0);
    rConsts.push(parseInt(document.getElementById(`dep_r_${k}_c`)?.value, 10) || 0);
  }

  // Tạo biểu thức chỉ số
  const makeIdx = (coeffs, cnst, labels) => {
    const parts = [];
    coeffs.forEach((a, j) => {
      if (a === 0) return;
      if (a === 1) parts.push(labels[j]);
      else if (a === -1) parts.push(`-${labels[j]}`);
      else parts.push(`${a}*${labels[j]}`);
    });
    if (cnst !== 0) parts.push(String(cnst));
    if (!parts.length) return "0";
    let s = parts[0];
    for (let i = 1; i < parts.length; i++)
      s += parts[i].startsWith("-") ? ` - ${parts[i].slice(1)}` : ` + ${parts[i]}`;
    return s;
  };

  const wExpr = Array.from({length: m}, (_, k) => makeIdx(wCoeffs[k], wConsts[k], iLabels));
  const rExpr = Array.from({length: m}, (_, k) => makeIdx(rCoeffs[k], rConsts[k], iLabels));
  const wArrStr = wExpr.map(esc).join("][");
  const rArrStr = rExpr.map(esc).join("][");

  // Mỗi dòng là một <div class="pline"> — đây là cách duy nhất để HTML xuống dòng đúng
  const line = (content, indent = 0) =>
    `<div class="pline">${"&nbsp;&nbsp;".repeat(indent)}${content}</div>`;
  const blank = () => `<div class="pline">&nbsp;</div>`;
  const sep   = () => `<div class="pline psep"></div>`;

  let html = "";

  // ── Phần 1: Vòng lặp Python ──────────────────────────────────
  for (let j = 0; j < d; j++) {
    const {L, U} = loops[j];
    html += line(
      `<span class="kw">for</span> ${iLabels[j]} <span class="kw">in</span> ` +
      `<span class="kw">range</span>(${L},&nbsp;${U + 1}):`,
      j
    );
  }

  // Dòng GHI (vàng, hover tooltip)
  html += line(
    `<span class="arr-w" title="GHI: ghi giá trị vào ô A[${wArrStr}]">` +
    `A[${wArrStr}]</span>&nbsp;<span class="kw">=</span>&nbsp;<span class="cmt">...</span>`,
    d
  );

  // Dòng ĐỌC (xanh lá, hover tooltip) — trên dòng riêng
  html += line(
    `<span class="cmt">...</span>&nbsp;` +
    `<span class="arr-r" title="ĐỌC: đọc giá trị từ ô A[${rArrStr}]">` +
    `A[${rArrStr}]</span>&nbsp;<span class="cmt">...</span>`,
    d
  );

  // ── Phần 2: Điều kiện phụ thuộc (tách biệt bằng đường kẻ) ────
  html += sep();
  html += line(`<span class="cmt"># Phụ thuộc tồn tại khi tìm được iw, ir thỏa:</span>`);
  blank();

  const iwLabels = Array.from({length: d}, (_, j) => `iw${j+1}`);
  const irLabels = Array.from({length: d}, (_, j) => `ir${j+1}`);

  // Phương trình bằng nhau — mỗi chiều 1 dòng
  for (let k = 0; k < m; k++) {
    const lhs = makeIdx(wCoeffs[k], wConsts[k], iwLabels);
    const rhs = makeIdx(rCoeffs[k], rConsts[k], irLabels);
    const note = m > 1 ? `&nbsp;&nbsp;<span class="cmt"># chiều ${k+1}</span>` : "";
    html += line(`<span class="eq">${esc(lhs)}&nbsp;==&nbsp;${esc(rhs)}</span>${note}`, 1);
  }

  html += blank();

  // Biên vòng lặp — iw và ir tách thành 2 dòng riêng cho mỗi vòng
  for (let j = 0; j < d; j++) {
    const {L, U} = loops[j];
    const note = d > 1 ? `&nbsp;&nbsp;<span class="cmt"># vòng ${j+1}</span>` : "";
    html += line(`<span class="eq">${L}&nbsp;&lt;=&nbsp;iw${j+1}&nbsp;&lt;=&nbsp;${U}</span>${note}`, 1);
    html += line(`<span class="eq">${L}&nbsp;&lt;=&nbsp;ir${j+1}&nbsp;&lt;=&nbsp;${U}</span>`, 1);
    if (j < d - 1) html += blank();
  }

  const el = document.getElementById("depEqPreview");
  if (el) el.innerHTML = html;
}

// Ví dụ mẫu
const DEP_EXAMPLES = [
  {
    name: "1 vòng, A[i] = A[i-1] (loop-carried dependence)",
    d: 1, m: 1,
    loops: [{L: 0, U: 10}],
    write: [[1, 0]],   // f(i) = 1·i + 0
    read:  [[1, -1]],  // g(i) = 1·i + (-1)
  },
  {
    name: "2 vòng lồng, A[i][j] = A[i-1][j+1]",
    d: 2, m: 2,
    loops: [{L: 0, U: 5}, {L: 0, U: 5}],
    write: [[1,0,0],[0,1,0]],   // f1=i1, f2=i2
    read:  [[1,0,-1],[0,1,1]],  // g1=i1-1, g2=i2+1
  },
  {
    name: "2 chiều mảng, A[i][2i] = A[i+1][2i-1]",
    d: 1, m: 2,
    loops: [{L: 0, U: 8}],
    write: [[1,0],[2,0]],    // f1=i, f2=2i
    read:  [[1,1],[2,-1]],   // g1=i+1, g2=2i-1
  },
  {
    name: "GCD loại: A[2i] vs A[2i+1] (chẵn ≠ lẻ)",
    d: 1, m: 1,
    loops: [{L: 0, U: 10}],
    write: [[2, 0]],   // f(i) = 2i
    read:  [[2, 1]],   // g(i) = 2i+1
  },
];

function depLoadExample(idx) {
  const ex = DEP_EXAMPLES[idx];
  if (!ex) return;
  depBuild(ex.d, ex.m);
  // Điền cận vòng lặp
  ex.loops.forEach((lp, j) => {
    document.getElementById(`dep_L${j}`).value = lp.L;
    document.getElementById(`dep_U${j}`).value = lp.U;
  });
  // Điền hệ số ghi
  ex.write.forEach((row, k) => {
    row.slice(0, ex.d).forEach((v, j) => {
      document.getElementById(`dep_w_${k}_${j}`).value = v;
    });
    document.getElementById(`dep_w_${k}_c`).value = row[ex.d] ?? 0;
  });
  // Điền hệ số đọc
  ex.read.forEach((row, k) => {
    row.slice(0, ex.d).forEach((v, j) => {
      document.getElementById(`dep_r_${k}_${j}`).value = v;
    });
    document.getElementById(`dep_r_${k}_c`).value = row[ex.d] ?? 0;
  });
  depUpdatePreview();
  document.getElementById("depResult").innerHTML = "";
}

// Thu thập dữ liệu từ form và gọi API
async function runDependGeneral() {
  const d = depD, m = depM;
  const loops = [];
  for (let j = 0; j < d; j++) {
    loops.push({
      L: parseInt(document.getElementById(`dep_L${j}`).value, 10) || 0,
      U: parseInt(document.getElementById(`dep_U${j}`).value, 10) || 0,
    });
  }
  const write = [], read = [];
  for (let k = 0; k < m; k++) {
    const wRow = [], rRow = [];
    for (let j = 0; j < d; j++) {
      wRow.push(parseInt(document.getElementById(`dep_w_${k}_${j}`).value, 10) || 0);
      rRow.push(parseInt(document.getElementById(`dep_r_${k}_${j}`).value, 10) || 0);
    }
    wRow.push(parseInt(document.getElementById(`dep_w_${k}_c`).value, 10) || 0);
    rRow.push(parseInt(document.getElementById(`dep_r_${k}_c`).value, 10) || 0);
    write.push(wRow);
    read.push(rRow);
  }

  const host = document.getElementById("depResult");
  host.innerHTML = `<div class="alert ok" style="opacity:.6">Đang kiểm tra…</div>`;

  try {
    const res = await fetch("/api/depend_general", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({d, loops, write, read}),
    });
    const data = await res.json();
    if (!res.ok) { host.innerHTML = `<div class="alert err"><span class="ic">⚠️</span>${esc(data.error)}</div>`; return; }
    renderDepResult(data, d);
  } catch (e) {
    host.innerHTML = `<div class="alert err"><span class="ic">⚠️</span>${esc(String(e))}</div>`;
  }
}

function renderDepResult(d, numVars) {
  const host = document.getElementById("depResult");

  // ── Lấy lại thông tin vòng lặp đang nhập để giải thích cụ thể ──
  const loops = [];
  for (let j = 0; j < depD; j++) {
    loops.push({
      L: parseInt(document.getElementById(`dep_L${j}`)?.value, 10) || 0,
      U: parseInt(document.getElementById(`dep_U${j}`)?.value, 10) || 0,
    });
  }
  const iLabels = Array.from({length: depD}, (_, j) => `i${j+1}`);
  const makeIdx = (coeffs, cnst, labels) => {
    const parts = [];
    coeffs.forEach((a, j) => {
      if (a === 0) return;
      if (a === 1) parts.push(labels[j]);
      else if (a === -1) parts.push(`-${labels[j]}`);
      else parts.push(`${a}·${labels[j]}`);
    });
    if (cnst !== 0) parts.push(String(cnst));
    return parts.length ? parts.join(" + ").replace(/\+ -/g, "- ") : "0";
  };
  const wIdxs = [], rIdxs = [];
  for (let k = 0; k < depM; k++) {
    const wRow = [], rRow = [];
    for (let j = 0; j < depD; j++) {
      wRow.push(parseInt(document.getElementById(`dep_w_${k}_${j}`)?.value, 10) || 0);
      rRow.push(parseInt(document.getElementById(`dep_r_${k}_${j}`)?.value, 10) || 0);
    }
    const cw = parseInt(document.getElementById(`dep_w_${k}_c`)?.value, 10) || 0;
    const cr = parseInt(document.getElementById(`dep_r_${k}_c`)?.value, 10) || 0;
    wIdxs.push(makeIdx(wRow, cw, iLabels));
    rIdxs.push(makeIdx(rRow, cr, iLabels));
  }
  const arrW = depM === 1 ? `A[${wIdxs[0]}]` : `A[${wIdxs.join("][")}]`;
  const arrR = depM === 1 ? `A[${rIdxs[0]}]` : `A[${rIdxs.join("][")}]`;
  const loopDesc = loops.map((lp, j) => `${iLabels[j]} ∈ [${lp.L}, ${lp.U}]`).join(", ");

  // ── Verdict ────────────────────────────────────────────────────
  let verdict = "";
  if (!d.has_dependence && d.gcd_eliminated) {
    verdict = `<div class="dep-verdict gcd">
      <span class="ic">✓</span>
      <div class="vbody">
        <div class="vtitle">KHÔNG có phụ thuộc — loại bởi GCD test</div>
        <div class="vdetail">
          Phương trình phụ thuộc <strong>không có nghiệm nguyên</strong>
          (điều kiện chia hết gcd không thỏa). Kết luận chính xác, không cần chạy FM.
        </div>
      </div>
    </div>`;
  } else if (!d.has_dependence) {
    verdict = `<div class="dep-verdict none">
      <span class="ic">✓</span>
      <div class="vbody">
        <div class="vtitle">KHÔNG có phụ thuộc</div>
        <div class="vdetail">
          Hệ ràng buộc <strong>vô nghiệm trên số thực</strong> ⟹ vô nghiệm trên số nguyên ⟹
          không bao giờ có hai lần lặp nào ghi và đọc trùng ô nhớ.
        </div>
      </div>
    </div>`;
  } else {
    const w = d.witness;
    const iwStr = w ? w.iw.map((v, j) => `${iLabels[j]}=${v}`).join(", ") : "";
    const irStr = w ? w.ir.map((v, j) => `${iLabels[j]}=${v}`).join(", ") : "";
    const witnessDetail = w
      ? `<div class="dep-witness">Nhân chứng: lần ghi (${esc(iwStr)}) và lần đọc (${esc(irStr)}) trỏ vào cùng ô nhớ</div>`
      : "";
    verdict = `<div class="dep-verdict has">
      <span class="ic">⚠️</span>
      <div class="vbody">
        <div class="vtitle">CÓ THỂ có phụ thuộc</div>
        <div class="vdetail">
          Hệ ràng buộc <strong>khả thi trên số thực</strong> — tồn tại cặp lần lặp
          có thể ghi và đọc trùng ô nhớ. Trình biên dịch phải giữ nguyên thứ tự.
        </div>
        ${witnessDetail}
      </div>
    </div>`;
  }

  // ── Giải thích ý nghĩa cụ thể với vòng lặp đang nhập ──────────
  let meaning = "";
  if (!d.has_dependence) {
    meaning = `<div class="dep-meaning">
      <div class="dep-meaning-title">Ý nghĩa với vòng lặp của bạn</div>
      <p>Với <code>${esc(loopDesc)}</code>:</p>
      <ul>
        <li>Lệnh ghi <code>${esc(arrW)} = …</code> và lệnh đọc <code>… ${esc(arrR)} …</code>
            <strong>không bao giờ trỏ vào cùng một ô nhớ</strong> trong suốt quá trình chạy.</li>
        <li>Trình biên dịch <strong>được phép</strong> chạy song song các lần lặp,
            đảo thứ tự, hoặc vectorize mà không làm sai kết quả.</li>
      </ul>
    </div>`;
  } else {
    const w = d.witness;
    let exampleLine = "";
    if (w) {
      const iwStr = w.iw.map((v, j) => `${iLabels[j]}=${v}`).join(", ");
      const irStr = w.ir.map((v, j) => `${iLabels[j]}=${v}`).join(", ");
      // Thay giá trị nhân chứng vào biểu thức chỉ số để hiện ô cụ thể
      const subVals = (expr, vals) => {
        let s = expr;
        iLabels.forEach((lbl, j) => { s = s.replace(new RegExp(lbl, "g"), vals[j]); });
        return s;
      };
      const wCell = depM === 1
        ? `A[${subVals(wIdxs[0], w.iw)}]`
        : `A[${wIdxs.map(e => subVals(e, w.iw)).join("][")}]`;
      const rCell = depM === 1
        ? `A[${subVals(rIdxs[0], w.ir)}]`
        : `A[${rIdxs.map(e => subVals(e, w.ir)).join("][")}]`;
      exampleLine = `<li>Ví dụ: lần lặp <strong>ghi</strong> (${esc(iwStr)}) ghi vào
          <code>${esc(wCell)}</code>; lần lặp <strong>đọc</strong> (${esc(irStr)}) đọc từ
          <code>${esc(rCell)}</code> — <strong>cùng một ô nhớ</strong>.</li>`;
    }
    const isRealOnly = w && w.iw.some(v => v.includes("/"));
    meaning = `<div class="dep-meaning dep-meaning-warn">
      <div class="dep-meaning-title">Ý nghĩa với vòng lặp của bạn</div>
      <p>Với <code>${esc(loopDesc)}</code>:</p>
      <ul>
        <li>Lệnh ghi <code>${esc(arrW)} = …</code> và lệnh đọc <code>… ${esc(arrR)} …</code>
            <strong>có thể trỏ vào cùng một ô nhớ</strong> ở hai lần lặp khác nhau.</li>
        ${exampleLine}
        <li>Nếu chạy song song, lần đọc có thể lấy giá trị <em>chưa được ghi</em>
            (hoặc đã bị ghi đè) → <strong>sai kết quả</strong>.</li>
        <li>Trình biên dịch <strong>phải giữ nguyên thứ tự</strong> thực thi.</li>
      </ul>
      ${isRealOnly ? `<p class="dep-note" style="margin-top:8px">
        Nhân chứng là số phân số (không nguyên) — phụ thuộc thực sự trên số nguyên
        cần kiểm tra thêm bằng Omega test. Kết luận hiện tại là bảo thủ (an toàn).
      </p>` : ""}
    </div>`;
  }

  // ── Trace (ẩn, bấm mới hiện) ───────────────────────────────────
  const trace = (d.trace && d.trace.items || []).map((it) => {
    const body = it.body ? `<div class="tr-body">${esc(it.body)}</div>` : "";
    const out = it.outcome ? `<div class="tr-out">⇒ ${esc(it.outcome)}</div>` : "";
    return `<div class="tr-item ${it.kind}"><div class="tr-h">
      <span class="tr-title">${esc(it.title)}</span></div>${body}${out}</div>`;
  }).join("");
  const traceSection = trace ? `
    <details class="dep-explain-wrap" style="margin-top:10px">
      <summary class="dep-explain-toggle">Xem chi tiết các bước khử Fourier–Motzkin ▾</summary>
      <div style="padding:10px 14px">${trace}</div>
    </details>` : "";

  host.innerHTML = verdict + meaning + traceSection;
}

// Khởi tạo form khi tải trang
document.addEventListener("DOMContentLoaded", () => {
  loadExample();
  depBuild(1, 1);
});
