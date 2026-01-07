import os
from typing import List, Tuple

import cv2
import fitz  # PyMuPDF
from reportlab.lib.colors import white, black
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from layout_utils import (
    compute_font_size,
    expand_box,
    keep_only_outer_boxes,
    resolve_label_conflicts,
    suppress_overlapping_boxes,
)
from onnx_model import DocLayoutModel
from paddle_ocr_engine import get_paddle_ocr


TextItem = Tuple[int, int, str, str, int]  # (y, x, text, label, box_height)
TextBox = Tuple[int, int, int, int]  # (x1, y1, x2, y2)


def draw_layout_boxes_and_ocr(
    image_path: str,
    output_dir: str,
    page_num: int | None = None,
) -> tuple[list[TextItem], list[TextBox]]:
    """
    Runs layout detection (ONNX) + OCR (PaddleOCR) on a single image.

    Returns:
        collected_text: list of (y, x, text, label, box_height)
        text_regions: list of (x1, y1, x2, y2)
    """
    os.makedirs(output_dir, exist_ok=True)

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError("❌ Image not found")

    model = DocLayoutModel.load_available()
    results = model.predict(image)

    vis = image.copy()
    collected_text: list[TextItem] = []
    text_regions: list[TextBox] = []
    ocr = get_paddle_ocr()

    for res in results:
        # 1) resolve label conflicts (same region, different labels)
        resolved_boxes = resolve_label_conflicts(res.boxes, iou_thresh=0.7)

        # 2) keep only outer boxes (drop small boxes inside big ones)
        outer_boxes = keep_only_outer_boxes(resolved_boxes)

        # 3) suppress overlaps
        filtered_boxes = suppress_overlapping_boxes(outer_boxes, iou_thresh=0.6)

        for box in filtered_boxes:
            x1, y1, x2, y2 = map(int, box.xyxy)
            label = res.names[int(box.cls)].lower()
            conf = box.conf

            # Draw layout boxes
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                vis,
                f"{label} {conf:.2f}",
                (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

            # Only OCR relevant text regions
            if label not in ["title", "plain text", "figure_caption"]:
                continue

            # Track regions for later masking in the PDF
            text_regions.append((x1, y1, x2, y2))

            H, W = image.shape[:2]
            x1, y1, x2, y2 = expand_box(
                x1,
                y1,
                x2,
                y2,
                img_w=W,
                img_h=H,
                pad_ratio=0.05,
                min_pad=10,
            )

            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            ocr_result = ocr.ocr(crop, cls=True)
            for block in ocr_result:
                if block is None:
                    continue
                for item in block:
                    if item is None or len(item) != 2:
                        continue
                    box_pts, (text, score) = item
                    if score < 0.6:
                        continue

                    tl_x, tl_y = box_pts[0]
                    abs_x = x1 + tl_x
                    abs_y = y1 + tl_y

                    # OCR box points → estimate height
                    ys = sorted([pt[1] for pt in box_pts])
                    raw_height = ys[-1] - ys[0]

                    # Clamp OCR height to avoid extreme outliers
                    box_height = min(max(raw_height, 8), 25)

                    collected_text.append(
                        (abs_y, abs_x, text, label, box_height),
                    )

    collected_text = _sort_multicolumn(collected_text)
    collected_text = _deduplicate_text(collected_text)

    # ================= SAVE OUTPUTS =================
    if page_num is not None:
        layout_path = os.path.join(output_dir, f"page_{page_num}_layout.jpg")
        text_path = os.path.join(output_dir, f"page_{page_num}_text.txt")
    else:
        layout_path = os.path.join(output_dir, "layout_output.jpg")
        text_path = os.path.join(output_dir, "extracted_text.txt")

    cv2.imwrite(layout_path, vis)
    with open(text_path, "w", encoding="utf-8") as f:
        for _, _, text, _, _ in collected_text:
            f.write(text.strip() + "\n")

    print("🖼 Layout saved:", layout_path)
    print("💾 Text saved:", text_path)

    return collected_text, text_regions


def _sort_multicolumn(text_items: List[TextItem], column_gap: int = 80) -> List[TextItem]:
    """
    Sort text in multi-column layout:
    1. group by x into columns
    2. sort inside each column by y
    """
    text_items = sorted(text_items, key=lambda x: x[1])  # sort by x
    columns: list[list[TextItem]] = []

    for item in text_items:
        placed = False
        for col in columns:
            if abs(item[1] - col[0][1]) < column_gap:
                col.append(item)
                placed = True
                break
        if not placed:
            columns.append([item])

    columns.sort(key=lambda c: c[0][1])
    sorted_text: list[TextItem] = []
    for col in columns:
        col.sort(key=lambda x: x[0])
        sorted_text.extend(col)

    return sorted_text


def _deduplicate_text(text_items: List[TextItem]) -> List[TextItem]:
    """
    Remove duplicate text lines while preserving order.
    """
    seen = set()
    final_items: list[TextItem] = []

    for item in text_items:
        _, _, text, _, _ = item
        clean = text.strip()
        if clean and clean not in seen:
            seen.add(clean)
            final_items.append(item)
    return final_items


def reconstruct_pdf_with_visible_text(
    image_path: str,
    text_items: List[TextItem],
    text_boxes: List[TextBox],
    pdf_path: str,
) -> None:
    """
    Creates a reconstructed PDF:
    - full image as background
    - masks text regions
    - draws visible selectable OCR text
    """
    img = cv2.imread(image_path)
    img_h, img_w = img.shape[:2]

    page_w, page_h = A4
    scale_x = page_w / img_w
    scale_y = page_h / img_h

    c = canvas.Canvas(pdf_path, pagesize=A4)

    # 1) Draw full image
    c.drawImage(
        ImageReader(image_path),
        0,
        0,
        width=page_w,
        height=page_h,
    )

    # 2) Mask text regions
    c.setFillColor(white)
    for x1, y1, x2, y2 in text_boxes:
        pdf_x = x1 * scale_x
        pdf_y = page_h - (y2 * scale_y)
        pdf_w = (x2 - x1) * scale_x
        pdf_h = (y2 - y1) * scale_y
        c.rect(pdf_x, pdf_y, pdf_w, pdf_h, fill=1, stroke=0)

    # 3) Draw selectable OCR text (layout-aware)
    c.setFillColor(black)
    for y, x, text, label, box_h in text_items:
        pdf_x = x * scale_x
        pdf_y = page_h - (y * scale_y)

        font_size = compute_font_size(box_h, scale_y)

        if label == "title":
            font_name = "Helvetica-Bold"
        elif label == "figure_caption":
            font_name = "Helvetica-Oblique"
        else:
            font_name = "Helvetica"

        c.setFont(font_name, font_size)
        c.drawString(pdf_x, pdf_y, text)

    c.save()


def pdf_to_images(pdf_path: str, dpi: int = 300) -> List[str]:
    """
    Convert PDF pages to images using PyMuPDF (NO poppler).
    Returns list of image file paths (one per page).
    """
    doc = fitz.open(pdf_path)

    image_paths: list[str] = []
    temp_dir = os.path.join(os.path.dirname(pdf_path), "temp_pages")
    os.makedirs(temp_dir, exist_ok=True)

    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    for page_num in range(len(doc)):
        page = doc[page_num]
        pix = page.get_pixmap(matrix=mat, alpha=False)

        img_path = os.path.join(temp_dir, f"page_{page_num + 1}.png")
        pix.save(img_path)
        image_paths.append(img_path)

    doc.close()
    return image_paths


def process_pdf_to_pdf(input_pdf: str, output_pdf: str, temp_output_dir: str) -> None:
    """
    Full PDF → PDF pipeline using PyMuPDF:
    - convert each page to an image
    - run layout + OCR
    - rebuild a selectable PDF with masked regions
    """
    os.makedirs(temp_output_dir, exist_ok=True)

    # 1) Convert PDF to images
    page_images = pdf_to_images(input_pdf)

    c = canvas.Canvas(output_pdf, pagesize=A4)

    for page_idx, img_path in enumerate(page_images):
        print(f"📄 Processing page {page_idx + 1}")

        # 2) Run existing IMAGE pipeline
        collected_text, text_regions = draw_layout_boxes_and_ocr(
            img_path,
            temp_output_dir,
            page_num=page_idx + 1,
        )

        img = cv2.imread(img_path)
        img_h, img_w = img.shape[:2]

        page_w, page_h = A4
        scale_x = page_w / img_w
        scale_y = page_h / img_h

        # 3) Draw page image
        c.drawImage(
            ImageReader(img_path),
            0,
            0,
            width=page_w,
            height=page_h,
        )

        # 4) Mask text regions
        c.setFillColor(white)
        for x1, y1, x2, y2 in text_regions:
            pdf_x = x1 * scale_x
            pdf_y = page_h - (y2 * scale_y)
            pdf_w = (x2 - x1) * scale_x
            pdf_h = (y2 - y1) * scale_y
            c.rect(pdf_x, pdf_y, pdf_w, pdf_h, fill=1, stroke=0)

        # 5) Draw selectable OCR text (layout-aware)
        c.setFillColor(black)
        for y, x, text, label, box_h in collected_text:
            pdf_x = x * scale_x
            pdf_y = page_h - (y * scale_y)

            font_size = compute_font_size(box_h, scale_y)

            if label == "title":
                font_name = "Helvetica-Bold"
            elif label == "figure_caption":
                font_name = "Helvetica-Oblique"
            else:
                font_name = "Helvetica"

            c.setFont(font_name, font_size)
            c.drawString(pdf_x, pdf_y, text)

        c.showPage()

    c.save()
    print("✅ Final selectable PDF saved:", output_pdf)


if __name__ == "__main__":
    # Small example – adjust paths or call from main.py instead.
    print(
        "This module defines the PDF <-> OCR pipeline. "
        "Use main.py to run it with your own paths.",
    )