/* ColorComic — Upload page logic */

const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const uploadBtn = document.getElementById('uploadBtn');
const fileInfo = document.getElementById('fileInfo');

let selectedFile = null;
let selectedRefFile = null;

// ── PDF Drag and Drop ───────────────────────────────────────────────────────

dropZone.addEventListener('click', () => fileInput.click());

dropZone.addEventListener('dragover', e => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', e => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length && files[0].name.toLowerCase().endsWith('.pdf')) {
        selectFile(files[0]);
    }
});

fileInput.addEventListener('change', () => {
    if (fileInput.files.length) selectFile(fileInput.files[0]);
});

function selectFile(file) {
    selectedFile = file;
    document.getElementById('fileName').textContent = file.name;
    const sizeMB = (file.size / (1024 * 1024)).toFixed(1);
    document.getElementById('fileSize').textContent = `(${sizeMB} MB)`;
    fileInfo.style.display = 'block';
    updateUploadButtonState();
}

// ── Colorization Mode Toggle ────────────────────────────────────────────────

const modeRadios = document.querySelectorAll('input[name="mode"]');
const referenceSection = document.getElementById('referenceSection');

modeRadios.forEach(radio => {
    radio.addEventListener('change', () => {
        const isReference = radio.value === 'reference' && radio.checked;
        referenceSection.style.display = isReference ? 'block' : 'none';
        updateUploadButtonState();
    });
});

function getSelectedMode() {
    const checked = document.querySelector('input[name="mode"]:checked');
    return checked ? checked.value : 'auto';
}

// ── Reference Image Upload ──────────────────────────────────────────────────

const refDropZone = document.getElementById('refDropZone');
const refFileInput = document.getElementById('refFileInput');
const refPlaceholder = document.getElementById('refPlaceholder');
const refPreview = document.getElementById('refPreview');
const refPreviewImg = document.getElementById('refPreviewImg');
const refRemoveBtn = document.getElementById('refRemoveBtn');

refDropZone.addEventListener('click', (e) => {
    if (e.target === refRemoveBtn || e.target.closest('#refRemoveBtn')) return;
    refFileInput.click();
});

refDropZone.addEventListener('dragover', e => {
    e.preventDefault();
    refDropZone.classList.add('dragover');
});

refDropZone.addEventListener('dragleave', () => {
    refDropZone.classList.remove('dragover');
});

refDropZone.addEventListener('drop', e => {
    e.preventDefault();
    refDropZone.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length && isImageFile(files[0])) {
        selectRefFile(files[0]);
    }
});

refFileInput.addEventListener('change', () => {
    if (refFileInput.files.length) selectRefFile(refFileInput.files[0]);
});

refRemoveBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    clearRefFile();
});

function isImageFile(file) {
    return /\.(png|jpe?g|webp)$/i.test(file.name);
}

function selectRefFile(file) {
    selectedRefFile = file;
    const url = URL.createObjectURL(file);
    refPreviewImg.src = url;
    refPlaceholder.style.display = 'none';
    refPreview.style.display = 'block';
    updateUploadButtonState();
}

function clearRefFile() {
    selectedRefFile = null;
    refFileInput.value = '';
    refPlaceholder.style.display = 'block';
    refPreview.style.display = 'none';
    if (refPreviewImg.src) {
        URL.revokeObjectURL(refPreviewImg.src);
        refPreviewImg.src = '';
    }
    updateUploadButtonState();
}

// ── Upload Button State ─────────────────────────────────────────────────────

function updateUploadButtonState() {
    const mode = getSelectedMode();
    if (!selectedFile) {
        uploadBtn.disabled = true;
        return;
    }
    if (mode === 'reference' && !selectedRefFile) {
        uploadBtn.disabled = true;
        return;
    }
    uploadBtn.disabled = false;
}

// ── GPU Detection ───────────────────────────────────────────────────────────

document.getElementById('detectGpuBtn').addEventListener('click', async () => {
    const btn = document.getElementById('detectGpuBtn');
    const status = document.getElementById('gpuDetectStatus');
    const infoBox = document.getElementById('gpuInfoBox');
    const gpuLabel = document.getElementById('gpuRadioLabel');

    btn.disabled = true;
    status.textContent = 'Detecting...';

    try {
        const res = await fetch('/api/gpu-info');
        const data = await res.json();

        if (!data.available) {
            infoBox.style.display = 'block';
            infoBox.innerHTML = '<strong>No GPU detected.</strong><br><span class="text-dim">CUDA is not available. Using CPU mode.</span>';
            status.textContent = '';
            btn.disabled = false;
            return;
        }

        const gpu = data.gpus[0];
        const recText = data.recommended === 'cuda'
            ? '<span style="color:#4caf50;">Recommended: GPU</span>'
            : '<span style="color:#ff9800;">Recommended: CPU</span> (low VRAM)';

        infoBox.style.display = 'block';
        infoBox.innerHTML = `
            <div style="display:flex; justify-content:space-between; flex-wrap:wrap; gap:0.5rem;">
                <div>
                    <strong>${gpu.name}</strong><br>
                    <span class="text-dim">VRAM:</span> ${gpu.vram_total_gb} GB total, ${gpu.vram_free_gb} GB free<br>
                    <span class="text-dim">Compute:</span> ${gpu.compute_capability} &middot; ${gpu.multi_processors} SMs<br>
                    <span class="text-dim">CUDA:</span> ${data.driver}
                </div>
                <div style="align-self:center;">${recText}</div>
            </div>
        `;

        // Show GPU radio and auto-select recommended
        gpuLabel.style.display = '';
        if (data.recommended === 'cuda') {
            document.querySelector('input[name="device"][value="cuda"]').checked = true;
        }

        status.textContent = '';
    } catch (err) {
        status.textContent = 'Detection failed';
        infoBox.style.display = 'block';
        infoBox.innerHTML = '<span class="text-dim">Could not detect GPU. Using CPU mode.</span>';
    }
    btn.disabled = false;
});

// ── Upload ──────────────────────────────────────────────────────────────────

uploadBtn.addEventListener('click', async () => {
    if (!selectedFile) return;
    uploadBtn.disabled = true;

    const formData = new FormData();
    formData.append('file', selectedFile);

    const mode = getSelectedMode();
    formData.append('mode', mode);

    if (mode === 'reference' && selectedRefFile) {
        formData.append('reference', selectedRefFile);
    }

    const style = document.querySelector('input[name="style"]:checked');
    if (style) formData.append('style', style.value);

    const device = document.querySelector('input[name="device"]:checked');
    if (device) formData.append('device', device.value);

    const progress = document.getElementById('uploadProgress');
    progress.style.display = 'block';
    document.getElementById('uploadStatus').textContent = 'Uploading PDF...';
    document.getElementById('uploadFill').style.width = '30%';

    try {
        const res = await fetch('/upload', { method: 'POST', body: formData });
        const data = await res.json();

        if (data.error) {
            document.getElementById('uploadStatus').textContent = 'Error: ' + data.error;
            uploadBtn.disabled = false;
            return;
        }

        document.getElementById('uploadFill').style.width = '60%';
        document.getElementById('uploadStatus').textContent = `Extracted ${data.page_count} pages. Starting colorization...`;

        // Start colorization
        await fetch(`/api/colorize/${data.job_id}`, { method: 'POST' });

        document.getElementById('uploadFill').style.width = '100%';

        // Redirect to processing page
        window.location.href = `/processing/${data.job_id}`;

    } catch (err) {
        document.getElementById('uploadStatus').textContent = 'Upload failed: ' + err.message;
        uploadBtn.disabled = false;
    }
});
