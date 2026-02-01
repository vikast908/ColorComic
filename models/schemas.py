from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class PanelRegion(BaseModel):
    """Detected panel within a page."""

    index: int
    x: int
    y: int
    width: int
    height: int


class JobState(BaseModel):
    """State of a colorization job."""

    job_id: str
    pdf_path: str
    page_count: int
    page_images: list[str] = []
    colorized_images: list[str] = []
    output_pdf: Optional[str] = None
    status: str = "uploaded"
    progress: float = 0.0
    current_step: str = ""
    style: str = "auto"
    device: str = "auto"
