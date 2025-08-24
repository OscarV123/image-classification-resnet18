// === CONFIG ===
const API_BASE    = "https://image-classification-resnet18-production.up.railway.app";
const PREDICT_URL = `${API_BASE}/predict`;

// === DOM ===
const dropzone    = document.getElementById("dropzone");
const fileInput   = document.getElementById("fileInput");
const folderInput = document.getElementById("folderInput");
const predList    = document.querySelector(".pred-list");
const searchInput = document.getElementById("q");
const btnPredict  = document.getElementById("btnPredict");
const btnClear    = document.getElementById("btnClear");
const dzThumbs    = document.getElementById("dzThumbs");
const botonesBar  = document.querySelector(".botones");

// === STORE GLOBAL ===
/** @type {{id:number,file:File,li:HTMLLIElement,status:'waiting'|'running'|'done'|'error'}[]} */
let items = [];
let nextId = 1;
const MAX_THUMBS = 12;

// ==== HELPERS DE ESTADO ====
function waitingCount() {
  return items.filter(it => it.status === "waiting").length;
}
function updatePredictState() {
  btnPredict.disabled = waitingCount() === 0;
}
function lockInputs() {
  fileInput.disabled = true;
  folderInput.disabled = true;
  dropzone.classList.add("disabled");
  botonesBar.classList.add("locked");
}
function unlockInputs() {
  fileInput.disabled = false;
  folderInput.disabled = false;
  dropzone.classList.remove("disabled");
  botonesBar.classList.remove("locked");
}

// ==== MANEJO DE ARCHIVOS ====
function handleFiles(fileList) {
  const files = Array.from(fileList);
  for (let i = files.length - 1; i >= 0; i--) {
    const file = files[i];
    if (!file.type.startsWith("image/")) continue;
    const li = renderListItem(file);       // UI derecha
    addThumb(file);                         // miniatura decorativa en dropzone
    items.push({ id: nextId++, file, li, status: "waiting" });
  }
  updateClearState();
  updatePredictState();
}

// Render de un item en la lista de resultados (estado Esperando)
function renderListItem(file) {
  const li = document.createElement("li");
  li.className = "pred-item";
  const url = URL.createObjectURL(file);
  li.innerHTML = `
    <img src="${url}" alt="miniatura ${file.name}" />
    <div class="pred-meta">
      <span class="pred-name">${file.name}</span>
      <span class="pred-class is-wait">Esperando</span>
    </div>
  `;
  predList.prepend(li);
  return li;
}

// Miniaturas en el dropzone
function addThumb(file) {
  const url = URL.createObjectURL(file);
  const img = document.createElement("img");
  img.className = "dz-thumb";
  img.src = url;
  img.alt = file.name;
  dzThumbs.prepend(img);

  while (dzThumbs.children.length > MAX_THUMBS) {
    const last = dzThumbs.lastElementChild;
    try { if (last.src.startsWith("blob:")) URL.revokeObjectURL(last.src); } catch {}
    last.remove();
  }
}

// ==== PREDICCIÓN ====
btnPredict.addEventListener("click", async() => {
  const batch = items.filter(it => it.status === "waiting");
  if (batch.length === 0) return;  // no hagas nada si no hay pendientes

  // bloquear y marcar "clasificando"
  lockInputs();
  btnPredict.disabled = true;
  for (const it of batch) {
    it.status = "running";
    const span = it.li.querySelector(".pred-class");
    span.textContent = "Clasificando…";
    span.classList.remove("is-wait");
    span.classList.add("is-run");
  }

  // enviar cada uno
  for (let i = batch.length - 1; i >= 0; i--) {
  const it = batch[i];
  await sendToBackend(it);
  await new Promise(r => setTimeout(r, 100));
}

  // No re-habilitamos inputs automáticamente (tu UX); se re-habilitan con "Limpiar"
  updatePredictState();
});

async function sendToBackend(item) {
  const span = item.li.querySelector(".pred-class");
  const formData = new FormData();
  formData.append("file", item.file);

  try {
    const res = await fetch(PREDICT_URL, { method: "POST", body: formData });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    const clase = data.prediction ?? "desconocido";
    let conf;
    if (data.probabilities && typeof data.probabilities === "object") {
      conf = data.probabilities[clase];
      if (conf == null) {
        const vals = Object.values(data.probabilities).filter(v => typeof v === "number");
        if (vals.length) conf = Math.max(...vals);
      }
    }
    const confPct = typeof conf === "number" ? Math.round(conf * 100) : 0;

    // color por rango
    let color = "#c22";                     // 0–50
    if (confPct > 80) color = "#2ea44f";    // 81–100
    else if (confPct > 50) color = "#e6c229"; // 51–80

    span.textContent = `${clase} (seguridad: ${confPct}%)`;
    span.style.color = color;
    span.classList.remove("is-run", "is-wait");
    item.status = "done";
  } catch (err) {
    console.error("Error:", err);
    span.textContent = "Error al clasificar";
    span.style.color = "#c22";
    span.classList.remove("is-run", "is-wait");
    item.status = "error";
  } finally {
    updateClearState();
  }
}

// ==== BÚSQUEDA ====
searchInput.addEventListener("input", e => {
  const q = e.target.value.toLowerCase();
  document.querySelectorAll(".pred-item").forEach(li => {
    const name = li.querySelector(".pred-name").textContent.toLowerCase();
    const pred = li.querySelector(".pred-class").textContent.toLowerCase();
    li.style.display = (name.includes(q) || pred.includes(q)) ? "" : "none";
  });
});

// ==== DRAG & DROP ====
["dragenter","dragover"].forEach(evt =>
  dropzone.addEventListener(evt, e => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  })
);
["dragleave","drop"].forEach(evt =>
  dropzone.addEventListener(evt, e => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
  })
);
dropzone.addEventListener("drop", e => handleFiles(e.dataTransfer.files));

// ==== INPUTS ====
fileInput.addEventListener("change", e => handleFiles(e.target.files));
folderInput.addEventListener("change", e => handleFiles(e.target.files));

// ==== CLEAR ====
function updateClearState() {
  btnClear.disabled = predList.querySelectorAll(".pred-item").length === 0;
}
updateClearState();

btnClear.addEventListener("click", () => {
  // liberar blobs de la lista
  predList.querySelectorAll(".pred-item img").forEach(img => {
    try { if (img.src.startsWith("blob:")) URL.revokeObjectURL(img.src); } catch {}
  });
  predList.innerHTML = "";

  // liberar blobs del dropzone
  dzThumbs.querySelectorAll("img").forEach(img => {
    try { if (img.src.startsWith("blob:")) URL.revokeObjectURL(img.src); } catch {}
    img.remove();
  });

  // resetear store y UI
  items = [];
  nextId = 1;
  updateClearState();
  unlockInputs();
  updatePredictState(); // sin pendientes → desactiva Predecir
});

// estado inicial
btnPredict.disabled = true;
