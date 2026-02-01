"""Comic panel detection and reading-order sorting."""

import cv2
import numpy as np

from models.schemas import PanelRegion


def detect_panels(
    page_image: np.ndarray, style: str = "western", min_area_ratio: float = 0.02
) -> list[PanelRegion]:
    """Detect rectangular panels in a comic page.

    Uses white-gap detection: finds horizontal and vertical white strips
    that divide the page into panels.
    """
    if len(page_image.shape) == 3:
        gray = cv2.cvtColor(page_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = page_image.copy()

    h, w = gray.shape[:2]
    page_area = h * w
    min_area = int(page_area * min_area_ratio)

    # --- Method: find white gaps (gutters) between panels ---
    # Threshold to find white areas
    _, white = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)

    # Project horizontally: a row is a "gap" if most pixels are white
    row_white = np.mean(white, axis=1) / 255.0
    # Project vertically: a column is a "gap" if most pixels are white
    col_white = np.mean(white, axis=0) / 255.0

    gap_threshold = 0.85  # 85% white = gutter
    min_gap = max(10, h // 100)  # minimum gap width in pixels

    h_splits = _find_splits(row_white, gap_threshold, min_gap, h)
    v_splits = _find_splits(col_white, gap_threshold, min_gap, w)

    # Add page boundaries
    h_bounds = [0] + h_splits + [h]
    v_bounds = [0] + v_splits + [w]

    panels = []
    for i in range(len(h_bounds) - 1):
        for j in range(len(v_bounds) - 1):
            y1, y2 = h_bounds[i], h_bounds[i + 1]
            x1, x2 = v_bounds[j], v_bounds[j + 1]
            pw, ph = x2 - x1, y2 - y1
            if pw * ph < min_area:
                continue
            # Skip if the region is nearly all white (it's a gutter)
            region = gray[y1:y2, x1:x2]
            if np.mean(region) > 240:
                continue
            panels.append(PanelRegion(index=0, x=x1, y=y1, width=pw, height=ph))

    # Fallback: if we found fewer than 2 panels, try contour-based method
    if len(panels) < 2:
        panels = _detect_panels_contour(gray, w, h, min_area)

    if not panels:
        panels = [PanelRegion(index=0, x=0, y=0, width=w, height=h)]

    panels = _sort_reading_order(panels, style)
    for i, p in enumerate(panels):
        p.index = i

    return panels


def _find_splits(profile: np.ndarray, threshold: float, min_gap: int, length: int) -> list[int]:
    """Find midpoints of white gap regions in a row/column projection."""
    in_gap = False
    gap_start = 0
    splits = []

    # Don't split at the very edges
    margin = length // 20

    for i in range(margin, length - margin):
        if profile[i] >= threshold:
            if not in_gap:
                gap_start = i
                in_gap = True
        else:
            if in_gap:
                gap_len = i - gap_start
                if gap_len >= min_gap:
                    splits.append((gap_start + i) // 2)
                in_gap = False

    return splits


def _detect_panels_contour(
    gray: np.ndarray, w: int, h: int, min_area: int
) -> list[PanelRegion]:
    """Fallback contour-based panel detection."""
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    inverted = cv2.bitwise_not(binary)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(inverted, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    panels = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, pw, ph = cv2.boundingRect(cnt)
        # Skip page border
        if pw > w * 0.95 and ph > h * 0.95:
            continue
        panels.append(PanelRegion(index=0, x=x, y=y, width=pw, height=ph))

    return panels


def _sort_reading_order(
    panels: list[PanelRegion], style: str, row_tolerance: float = 0.3
) -> list[PanelRegion]:
    """Sort panels into reading order by clustering rows then sorting within."""
    if len(panels) <= 1:
        return panels

    avg_height = sum(p.height for p in panels) / len(panels)
    tolerance = avg_height * row_tolerance

    sorted_by_y = sorted(panels, key=lambda p: p.y)

    rows: list[list[PanelRegion]] = []
    current_row: list[PanelRegion] = [sorted_by_y[0]]

    for panel in sorted_by_y[1:]:
        if abs(panel.y - current_row[0].y) <= tolerance:
            current_row.append(panel)
        else:
            rows.append(current_row)
            current_row = [panel]
    rows.append(current_row)

    reverse = style == "manga"
    ordered = []
    for row in rows:
        row.sort(key=lambda p: p.x, reverse=reverse)
        ordered.extend(row)

    return ordered


def extract_panel_image(
    page_image: np.ndarray, panel: PanelRegion
) -> np.ndarray:
    """Crop page image to a single panel's bounding box."""
    return page_image[
        panel.y : panel.y + panel.height, panel.x : panel.x + panel.width
    ]
