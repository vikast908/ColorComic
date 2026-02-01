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
