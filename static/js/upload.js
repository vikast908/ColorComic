/* ColorComic — Upload page logic */

const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const uploadBtn = document.getElementById('uploadBtn');
const fileInfo = document.getElementById('fileInfo');

let selectedFile = null;

// Drag and drop
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
    uploadBtn.disabled = false;
}

// GPU detection
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

// Upload
uploadBtn.addEventListener('click', async () => {
    if (!selectedFile) return;
    uploadBtn.disabled = true;

    const formData = new FormData();
    formData.append('file', selectedFile);

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
