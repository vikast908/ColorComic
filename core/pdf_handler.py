"""PDF extraction and reassembly using PyMuPDF."""

import os

import cv2
import fitz  # PyMuPDF


def get_page_count(pdf_path: str) -> int:
    with fitz.open(pdf_path) as doc:
        return len(doc)


def extract_pages(pdf_path: str, output_dir: str, dpi: int = 300) -> list[str]:
    """Render each PDF page as a PNG image. Returns list of image paths."""
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    with fitz.open(pdf_path) as doc:
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        for i, page in enumerate(doc):
            pix = page.get_pixmap(matrix=mat)
            out_path = os.path.join(output_dir, f"page_{i:04d}.png")
            pix.save(out_path)
            paths.append(out_path)
    return paths


def get_page_image_bytes(pdf_path: str, page_num: int, dpi: int = 150) -> bytes:
    """Return a single page as PNG bytes."""
    with fitz.open(pdf_path) as doc:
        page = doc[page_num]
        zoom = dpi / 72.0
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        return pix.tobytes("png")


def reassemble_pdf(
    image_paths: list[str],
    output_path: str,
    original_pdf_path: str | None = None,
) -> str:
    """Combine colorized page images into a new PDF.

    If original_pdf_path is given, match each page's dimensions to the original.
    """
    doc = fitz.open()

    orig_doc = None
    if original_pdf_path:
        orig_doc = fitz.open(original_pdf_path)

    for i, img_path in enumerate(image_paths):
        if orig_doc and i < len(orig_doc):
            orig_page = orig_doc[i]
            w, h = orig_page.rect.width, orig_page.rect.height
        else:
            # Use cv2 to get image dimensions (fast, no PDF overhead)
            img = cv2.imread(img_path)
            if img is not None:
                ih, iw = img.shape[:2]
                # Convert pixels to points at 72 DPI
                w, h = float(iw), float(ih)
                del img
            else:
                w, h = 612.0, 792.0  # US Letter fallback

        page = doc.new_page(width=w, height=h)
        page.insert_image(page.rect, filename=img_path)

    if orig_doc:
        orig_doc.close()

    # Skip deflate — images are already JPEG compressed
    doc.save(output_path)
    doc.close()
    return output_path
