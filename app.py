"""ColorComic — Flask application for B&W comic PDF colorization."""

import json
import os
import queue
import threading
import uuid

import cv2
import torch
from dotenv import load_dotenv
from flask import (
    Flask,
    Response,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)

from config import Config
from core.color_consistency import ColorConsistencyManager
from core.model_manager import ModelManager
from core.model_downloader import ensure_models_downloaded
from core.pdf_handler import extract_pages, get_page_count, reassemble_pdf
from core.postprocessor import PostProcessor
from core.upscaler import Upscaler
from models.schemas import JobState

load_dotenv()

# Enable cudnn autotuner for consistent input sizes (mc-v2 576x576, MangaNinja 512x512)
torch.backends.cudnn.benchmark = True

app = Flask(__name__)
app.config.from_object(Config)
app.secret_key = Config.SECRET_KEY

# In-memory job store (single-user / local use)
jobs: dict[str, JobState] = {}
job_queues: dict[str, queue.Queue] = {}

# Ensure directories exist
for folder in (Config.UPLOAD_FOLDER, Config.OUTPUT_FOLDER):
    os.makedirs(folder, exist_ok=True)

# ── Load ML model at startup ────────────────────────────────────────────────

print("Checking model weights...")
ensure_models_downloaded(Config.WEIGHTS_DIR, callback=print)

print("Initializing model manager...")
model_manager = ModelManager(device=Config.ML_DEVICE)

# Pre-load auto mode (mc-v2) at startup
model_manager.get_colorizer("auto")

# ── Post-processing pipeline ────────────────────────────────────────────────

upscaler = Upscaler(
    model_path=Config.ESRGAN_MODEL_PATH,
    model_url=Config.ESRGAN_MODEL_URL,
    scale=Config.ESRGAN_SCALE,
    tile=Config.ESRGAN_TILE,
    device=Config.ML_DEVICE,
) if Config.POSTPROCESS_UPSCALE else None

post_processor = PostProcessor(
    l_channel=Config.POSTPROCESS_L_CHANNEL,
    guided_filter=Config.POSTPROCESS_GUIDED_FILTER,
    upscale=Config.POSTPROCESS_UPSCALE,
    upscaler=upscaler,
)


# ── Pages ────────────────────────────────────────────────────────────────────


@app.route("/")
def index():
    return render_template(
        "index.html",
        cuda_available=model_manager.cuda_available,
        current_device=model_manager.device_name,
    )


@app.route("/preview/<job_id>")
def preview_view(job_id):
    job = jobs.get(job_id)
    if not job:
        return redirect(url_for("index"))
    return render_template("preview.html", job=job)


@app.route("/processing/<job_id>")
def processing_view(job_id):
    job = jobs.get(job_id)
    if not job:
        return redirect(url_for("index"))
    return render_template("processing.html", job=job)


# ── API: Upload ──────────────────────────────────────────────────────────────


@app.route("/upload", methods=["POST"])
def upload_pdf():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    f = request.files["file"]
    if not f.filename or not f.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are accepted"}), 400

    job_id = str(uuid.uuid4())[:12]
    job_dir = os.path.join(Config.UPLOAD_FOLDER, job_id)
    os.makedirs(job_dir, exist_ok=True)

    pdf_path = os.path.join(job_dir, f.filename)
    f.save(pdf_path)

    page_count = get_page_count(pdf_path)
    pages_dir = os.path.join(job_dir, "pages")
    page_images = extract_pages(pdf_path, pages_dir, dpi=Config.PAGE_DPI)

    style = request.form.get("style", "auto")
    device = request.form.get("device", "auto")
    mode = request.form.get("mode", "auto")

    # Handle reference image for reference mode
    reference_image_path = None
    if mode == "reference" and "reference" in request.files:
        ref_file = request.files["reference"]
        if ref_file.filename:
            ref_path = os.path.join(job_dir, "reference" + os.path.splitext(ref_file.filename)[1])
            ref_file.save(ref_path)
            reference_image_path = ref_path

    if mode == "reference" and not reference_image_path:
        return jsonify({"error": "Reference mode requires a reference image"}), 400

    job = JobState(
        job_id=job_id,
        pdf_path=pdf_path,
        page_count=page_count,
        page_images=page_images,
        style=style,
        device=device,
        mode=mode,
        reference_image_path=reference_image_path,
    )
    jobs[job_id] = job

    return jsonify({"job_id": job_id, "page_count": page_count})


# ── API: Serve page images ──────────────────────────────────────────────────


@app.route("/pages/<job_id>/<int:page_num>")
def serve_page(job_id, page_num):
    job = jobs.get(job_id)
    if not job or page_num < 0 or page_num >= len(job.page_images):
        return "Not found", 404
    return send_file(job.page_images[page_num], mimetype="image/png")


# ── API: Preview (serve pre-computed colorized images) ───────────────────────


@app.route("/api/preview/<job_id>/<int:page_num>")
def get_preview(job_id, page_num):
    job = jobs.get(job_id)
    if not job:
        return "Not found", 404
    if page_num < 0 or page_num >= len(job.colorized_images):
        return "Page not colorized yet", 400
    path = job.colorized_images[page_num]
    mime = "image/jpeg" if path.lower().endswith(".jpg") else "image/png"
    return send_file(path, mimetype=mime)


# ── API: Colorize (ML pipeline) ─────────────────────────────────────────────


@app.route("/api/colorize/<job_id>", methods=["POST"])
def start_colorize(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404

    job.status = "colorizing"
    job.progress = 0.0
    q = queue.Queue()
    job_queues[job_id] = q

    out_dir = os.path.join(Config.OUTPUT_FOLDER, job_id)
    os.makedirs(out_dir, exist_ok=True)

    def _run():
        try:
            # Switch device if user requested a specific one
            model_manager.switch_device(job.device)

            # Get the right colorizer for this mode
            colorizer = model_manager.get_colorizer(job.mode)

            # Load reference image once outside the loop
            ref_image = None
            if job.mode == "reference" and job.reference_image_path:
                ref_image = cv2.imread(job.reference_image_path)

            consistency = ColorConsistencyManager()
            colored_paths = []

            with torch.inference_mode():
                for i, img_path in enumerate(job.page_images):
                    q.put({"page": i, "total": job.page_count, "status": "colorizing"})

                    image = cv2.imread(img_path)

                    # Colorize based on mode
                    if job.mode == "reference":
                        result = colorizer.colorize(image, reference_image=ref_image)
                    else:
                        result = colorizer.colorize(image)

                    # Post-processing (applied to both modes)
                    result = post_processor.process(result, image)

                    # Color consistency: only for auto mode (reference mode has
                    # inherent consistency from the reference image)
                    if job.mode == "auto":
                        if i == 0:
                            consistency.set_reference(result)
                        else:
                            result = consistency.apply(
                                result, strength=Config.COLOR_TRANSFER_STRENGTH
                            )

                    out_path = os.path.join(out_dir, f"colored_{i:04d}.jpg")
                    cv2.imwrite(out_path, result, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    colored_paths.append(out_path)

                    # Update incrementally so preview endpoint works during processing
                    job.colorized_images.append(out_path)

                    job.progress = (i + 1) / job.page_count
                    q.put({"page": i, "total": job.page_count, "status": "done_page"})

            # Reassemble PDF
            output_pdf = os.path.join(out_dir, "colorized.pdf")
            reassemble_pdf(colored_paths, output_pdf, job.pdf_path)
            job.output_pdf = output_pdf
            job.status = "done"
            q.put({"done": True})
        except Exception as e:
            job.status = "error"
            job.current_step = str(e)
            q.put({"error": str(e), "done": True})
        finally:
            job_queues.pop(job_id, None)

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"ok": True})


@app.route("/api/colorize/<job_id>/stream")
def stream_colorize(job_id):
    def generate():
        q = job_queues.get(job_id)
        if not q:
            yield f"data: {json.dumps({'error': 'No active job', 'done': True})}\n\n"
            return
        while True:
            try:
                event = q.get(timeout=30)
                yield f"data: {json.dumps(event)}\n\n"
                if event.get("done"):
                    break
            except queue.Empty:
                yield f"data: {json.dumps({'heartbeat': True})}\n\n"

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── API: Download ────────────────────────────────────────────────────────────


@app.route("/api/download/<job_id>")
def download_pdf(job_id):
    job = jobs.get(job_id)
    if not job or not job.output_pdf:
        return "Not ready", 404
    return send_file(job.output_pdf, as_attachment=True, download_name="colorized.pdf")


# ── API: Status ──────────────────────────────────────────────────────────────


@app.route("/api/status")
def model_status():
    return jsonify({
        "model_loaded": True,
        "device": model_manager.device_name,
        "cuda_available": model_manager.cuda_available,
        "current_mode": model_manager.current_mode,
    })


@app.route("/api/gpu-info")
def gpu_info():
    """Return detailed GPU information for the user to review."""
    if not torch.cuda.is_available():
        return jsonify({"available": False})

    gpu_count = torch.cuda.device_count()
    gpus = []
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        mem_total = round(props.total_memory / (1024 ** 3), 1)
        mem_used = round(torch.cuda.memory_allocated(i) / (1024 ** 3), 2)
        mem_free = round(mem_total - mem_used, 1)
        gpus.append({
            "index": i,
            "name": props.name,
            "vram_total_gb": mem_total,
            "vram_free_gb": mem_free,
            "compute_capability": f"{props.major}.{props.minor}",
            "multi_processors": props.multi_processor_count,
        })

    recommended = "cuda" if gpus and gpus[0]["vram_total_gb"] >= 2 else "cpu"

    return jsonify({
        "available": True,
        "driver": torch.version.cuda,
        "gpu_count": gpu_count,
        "gpus": gpus,
        "recommended": recommended,
    })


# ── Run ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=5000, threaded=True, use_reloader=False)
