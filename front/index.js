const API_BASE = "https://image-classification-resnet18-production.up.railway.app";
const PREDICT_URL = `${API_BASE}/predict`;

const dropzone    = document.getElementById("dropzone");
const fileInput   = document.getElementById("fileInput");
const folderInput = document.getElementById("folderInput");
const predList    = document.querySelector(".pred-list");
const searchInput = document.getElementById("q");

function handleFiles(files) {
  [...files].forEach(file => {
    if (!file.type.startsWith("image/")) return;
    const li = addPreview(file);
    addThumb(file);
    sendToBackend(file, li);
  });
}

async function sendToBackend(file, li) {
  const span = li.querySelector(".pred-class");
  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch(PREDICT_URL, { method: "POST", body: formData });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);

    const data = await res.json();

    const clase = data.prediction ?? "desconocido";
    let conf = undefined;

    if (data.probabilities && typeof data.probabilities === "object") {
      conf = data.probabilities[clase];
      if (conf == null) {
        const vals = Object.values(data.probabilities).filter(v => typeof v === "number");
        if (vals.length) conf = Math.max(...vals);
      }
    }

    const confPct = conf ? Math.round(conf * 100) : 0;

    let color = "#c22";
    if (confPct > 80) {
      color = "#2ea44f";
    } else if (confPct > 50) {
      color = "#e6c229";
    }

    span.textContent = `${clase} (seguridad: ${confPct}%)`;
    span.style.color = color;

  } catch (err) {
    console.error("Error:", err);
    span.textContent = "Error al clasificar";
    span.style.color = "#c22";
  }
}

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

fileInput.addEventListener("change", e => handleFiles(e.target.files));
folderInput.addEventListener("change", e => handleFiles(e.target.files));

searchInput.addEventListener("input", e => {
  const q = e.target.value.toLowerCase();
  document.querySelectorAll(".pred-item").forEach(li => {
    const name = li.querySelector(".pred-name").textContent.toLowerCase();
    const pred = li.querySelector(".pred-class").textContent.toLowerCase();
    li.style.display = (name.includes(q) || pred.includes(q)) ? "" : "none";
  });
});

const btnClear = document.getElementById("btnClear");

function updateClearState() {
  btnClear.disabled = predList.querySelectorAll(".pred-item").length === 0;
}
updateClearState();

function addPreview(file) {
  const li = document.createElement("li");
  li.className = "pred-item";
  li.innerHTML = `
    <img src="${URL.createObjectURL(file)}" alt="miniatura ${file.name}" />
    <div class="pred-meta">
      <span class="pred-name">${file.name}</span>
      <span class="pred-class">Clasificando...</span>
    </div>
  `;
  predList.prepend(li);
  updateClearState();
  return li;
}

btnClear.addEventListener("click", () => {
  predList.querySelectorAll(".pred-item img").forEach(img => {
    try { if (img.src.startsWith("blob:")) URL.revokeObjectURL(img.src); } catch {}
  });
  predList.innerHTML = "";
  updateClearState();
});

btnClear.addEventListener("click", () => {
  predList.querySelectorAll(".pred-item img").forEach(img => {
    try { if (img.src.startsWith("blob:")) URL.revokeObjectURL(img.src); } catch {}
  });
  predList.innerHTML = "";
  updateClearState();

  dzThumbs.querySelectorAll("img").forEach(img => {
    try { if (img.src.startsWith("blob:")) URL.revokeObjectURL(img.src); } catch {}
    img.remove();
  });
});

const dzThumbs = document.getElementById("dzThumbs");
const MAX_THUMBS = 12;

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
