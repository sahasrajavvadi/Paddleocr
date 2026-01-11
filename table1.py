import abc
import os
import os.path
import cv2
import numpy as np
import ast
import fitz  # PyMuPDF (NO poppler needed)
from paddleocr import PPStructure
from bs4 import BeautifulSoup


from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.lib.colors import white, black


# ================= ONNX IMPORT =================
try:
    import onnx
    import onnxruntime
except ImportError as e:
    if "DLL load failed" in str(e):
        raise OSError(
            "Microsoft Visual C++ Redistributable is not installed. "
            "Download it at https://aka.ms/vs/17/release/vc_redist.x64.exe"
        ) from e
    raise



# ================= PADDLE OCR (OFFLINE) =================
from paddleocr import PaddleOCR

ocr = PaddleOCR(
    lang='en',
    use_angle_cls=True,
    det_model_dir=r"C:\Users\sahas\Downloads\paddleocrfinal\paddleocrmodels\det",
    rec_model_dir=r"C:\Users\sahas\Downloads\paddleocrfinal\paddleocrmodels\rec",
    cls_model_dir=r"C:\Users\sahas\Downloads\paddleocrfinal\paddleocrmodels\cls",
    use_gpu=False
)
table_engine = PPStructure(
    layout=False,   # ❗ YOLO already does layout
    table=True,
    ocr=True,
    show_log=False,
    use_gpu=False
)


print("✅ PaddleOCR ready (OFFLINE)")

# ================= BASE MODEL =================
class DocLayoutModel(abc.ABC):
    @staticmethod
    def load_onnx():
        model = OnnxModel.from_pretrained()
        return model

    @staticmethod
    def load_available():
        return DocLayoutModel.load_onnx()

    @property
    @abc.abstractmethod
    def stride(self) -> int:
        pass

    @abc.abstractmethod
    def predict(self, image, imgsz=1024, **kwargs) -> list:
        pass

# ================= YOLO RESULT =================
class YoloResult:
    def __init__(self, boxes, names):
        self.boxes = [YoloBox(data=d) for d in boxes]
        self.boxes.sort(key=lambda x: x.conf, reverse=True)
        self.names = names

class YoloBox:
    def __init__(self, data):
        self.xyxy = data[:4]
        self.conf = data[-2]
        self.cls = data[-1]

# ================= ONNX MODEL =================
class OnnxModel(DocLayoutModel):
    def __init__(self, model_path: str):
        self.model_path = model_path

        model = onnx.load(model_path)
        metadata = {d.key: d.value for d in model.metadata_props}
        self._stride = ast.literal_eval(metadata["stride"])
        self._names = ast.literal_eval(metadata["names"])

        self.model = onnxruntime.InferenceSession(
            model.SerializeToString(),
            providers=["CPUExecutionProvider"]
        )

    @staticmethod
    def from_pretrained():
        pth = r"C:\Users\sahas\Downloads\pdfocr\doclayout_yolo_docstructbench_imgsz1024.onnx"
        return OnnxModel(pth)

    @property
    def stride(self):
        return self._stride

    def resize_and_pad_image(self, image, new_shape):
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        h, w = image.shape[:2]
        new_h, new_w = new_shape

        r = min(new_h / h, new_w / w)
        resized_h, resized_w = int(round(h * r)), int(round(w * r))

        image = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

        pad_w = (new_w - resized_w) % self.stride
        pad_h = (new_h - resized_h) % self.stride
        top, bottom = pad_h // 2, pad_h - pad_h // 2
        left, right = pad_w // 2, pad_w - pad_w // 2

        image = cv2.copyMakeBorder(
            image, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        return image

    def scale_boxes(self, img1_shape, boxes, img0_shape):
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad_x = round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1)
        pad_y = round((img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1)

        boxes[..., :4] = (boxes[..., :4] - [pad_x, pad_y, pad_x, pad_y]) / gain
        return boxes

    def predict(self, image, imgsz=1024, **kwargs):
        orig_h, orig_w = image.shape[:2]
        pix = self.resize_and_pad_image(image, imgsz)
        pix = np.transpose(pix, (2, 0, 1))
        pix = np.expand_dims(pix, axis=0)
        pix = pix.astype(np.float32) / 255.0

        new_h, new_w = pix.shape[2:]
        preds = self.model.run(None, {"images": pix})[0]

        # IMPORTANT: lower threshold
        preds = preds[preds[..., 4] > 0.10]
        preds[..., :4] = self.scale_boxes(
            (new_h, new_w), preds[..., :4], (orig_h, orig_w)
        )

        return [YoloResult(boxes=preds, names=self._names)]
    
# ================= BOX EXPANSION (ADD ONLY) =================
def expand_box(x1, y1, x2, y2, img_w, img_h, pad_ratio=0.05, min_pad=10):
    """
    Expand box by a percentage + minimum pixels
    Prevents top/bottom text from getting cut
    """
    box_w = x2 - x1
    box_h = y2 - y1

    pad_x = max(int(box_w * pad_ratio), min_pad)
    pad_y = max(int(box_h * pad_ratio), min_pad)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(img_w, x2 + pad_x)
    y2 = min(img_h, y2 + pad_y)

    return x1, y1, x2, y2

# ================= LAYOUT NMS (ADD ONLY) =================
def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - inter + 1e-6
    return inter / union


def is_inside(inner, outer, margin=5):
    """
    Check if inner box is fully inside outer box
    inner, outer = (x1, y1, x2, y2)
    """
    return (
        inner[0] >= outer[0] - margin and
        inner[1] >= outer[1] - margin and
        inner[2] <= outer[2] + margin and
        inner[3] <= outer[3] + margin
    )


def resolve_label_conflicts(boxes, iou_thresh=0.7):
    """
    If multiple labels are predicted for the same region,
    keep ONLY the box with highest confidence.
    """
    boxes = sorted(boxes, key=lambda b: b.conf, reverse=True)
    final_boxes = []

    for box in boxes:
        keep = True
        for kept in final_boxes:
            iou = compute_iou(box.xyxy, kept.xyxy)

            # Same physical region → discard lower-confidence one
            if iou > iou_thresh:
                keep = False
                break

        if keep:
            final_boxes.append(box)

    return final_boxes




def suppress_overlapping_boxes(boxes, iou_thresh=0.6):
    """
    Removes:
    1) Highly overlapping boxes (same class)
    2) Small boxes fully inside bigger boxes (same class)
    """
    boxes = sorted(
        boxes,
        key=lambda b: (
            (b.xyxy[2] - b.xyxy[0]) * (b.xyxy[3] - b.xyxy[1]),  # area
            b.conf
        ),
        reverse=True
    )

    kept = []

    for box in boxes:
        x1, y1, x2, y2 = box.xyxy
        area = (x2 - x1) * (y2 - y1)

        discard = False
        for kept_box in kept:
            if box.cls != kept_box.cls:
                continue

            # 1️⃣ If box is fully inside a bigger box → DROP
            if is_inside(box.xyxy, kept_box.xyxy):
                discard = True
                break

            # 2️⃣ If heavy overlap → DROP
            iou = compute_iou(box.xyxy, kept_box.xyxy)
            if iou > iou_thresh:
                discard = True
                break

        if not discard:
            kept.append(box)

    return kept

def keep_only_outer_boxes(boxes, margin=5):
    """
    If a box is fully inside another bigger box,
    keep ONLY the bigger box (regardless of label).
    """
    boxes = sorted(
        boxes,
        key=lambda b: (b.xyxy[2] - b.xyxy[0]) * (b.xyxy[3] - b.xyxy[1]),
        reverse=True
    )

    kept = []

    for box in boxes:
        discard = False
        for big in kept:
            if is_inside(box.xyxy, big.xyxy, margin=margin):
                discard = True
                break
        if not discard:
            kept.append(box)

    return kept


# ================= FONT SIZE UTILITY =================
def compute_font_size(box_height_px, scale_y, min_size=6, max_size=18):
    font_size = box_height_px * scale_y * 0.75
    return max(min_size, min(font_size, max_size))


import csv

def save_ppstructure_table(table_output, output_dir, page_num, table_idx):
    from bs4 import BeautifulSoup

    html = None
    for item in table_output:
        if item["type"] == "table":
            html = item["res"]["html"]
            break

    if html is None:
        return

    soup = BeautifulSoup(html, "html.parser")
    rows = []

    for tr in soup.find_all("tr"):
        row = [td.get_text(strip=True) for td in tr.find_all("td")]
        if row:
            rows.append(row)

    csv_path = os.path.join(
        output_dir,
        f"page_{page_num}_table_{table_idx}.csv"
    )

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print("📊 Table CSV saved:", csv_path)



# ================= DRIVER WITH OCR =================
def draw_layout_boxes_and_ocr(image_path, output_dir, page_num=None):
    os.makedirs(output_dir, exist_ok=True)

    table_results = []   # ✅ store PP-Structure tables

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError("❌ Image not found")

    model = DocLayoutModel.load_available()
    results = model.predict(image)

    vis = image.copy()
    collected_text = []
    final_boxes = []

    for res in results:
        resolved_boxes = resolve_label_conflicts(res.boxes, iou_thresh=0.7)
        outer_boxes = keep_only_outer_boxes(resolved_boxes)
        filtered_boxes = suppress_overlapping_boxes(outer_boxes, iou_thresh=0.6)

        final_boxes.extend(filtered_boxes)

        for box in filtered_boxes:
            x1, y1, x2, y2 = map(int, box.xyxy)
            label = res.names[int(box.cls)].lower()

            # Draw layout boxes
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                vis, label,
                (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 0), 1
            )


            if label not in ["title", "plain text", "figure_caption", "table", "table_footnote","table_caption"]:
                continue

            H, W = image.shape[:2]
            x1, y1, x2, y2 = expand_box(x1, y1, x2, y2, W, H)

            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # ================= TABLE → PP-STRUCTURE =================
            if label == "table":
                # 1️⃣ OCR TABLE TEXT WITH REAL COORDINATES
               
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

                        ys = [pt[1] for pt in box_pts]
                        box_height = min(max(max(ys) - min(ys), 8), 25)

                        collected_text.append(
                            (abs_y, abs_x, text, "table", box_height)
                        )

                # 2️⃣ PP-STRUCTURE ONLY FOR CSV
                table_output = table_engine(crop)
                if table_output:
                    table_results.append({
                        "page": page_num,
                        "bbox": (x1, y1, x2, y2),
                        "result": table_output
                    })

                continue

            # ================= TEXT → PADDLE OCR =================
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

                    ys = [pt[1] for pt in box_pts]
                    box_height = min(max(max(ys) - min(ys), 8), 25)

                    collected_text.append(
                        (abs_y, abs_x, text, label, box_height)
                    )

    # ================= ADD TABLE TEXT INTO OCR PIPELINE =================
    

    # ================= SORT TEXT =================
    collected_text = sorted(collected_text, key=lambda x: (x[0], x[1]))

    # ================= TEXT REGIONS =================
    text_regions = []
    for box in final_boxes:
        label = res.names[int(box.cls)].lower()
        if label in ["title", "plain text", "figure_caption", "table","table_footnote","table_caption"]:
            x1, y1, x2, y2 = map(int, box.xyxy)
            text_regions.append((x1, y1, x2, y2))

    # ================= SAVE TABLE CSVs =================
    for idx, table in enumerate(table_results, start=1):
        save_ppstructure_table(
            table["result"],
            output_dir,
            page_num,
            idx
        )

    # ================= SAVE OUTPUT FILES =================
    layout_path = os.path.join(output_dir, f"page_{page_num}_layout.jpg")
    text_path = os.path.join(output_dir, f"page_{page_num}_text.txt")

    cv2.imwrite(layout_path, vis)
    with open(text_path, "w", encoding="utf-8") as f:
        f.write("\n".join(t[2] for t in collected_text if t[2]))

    print("🖼 Layout saved:", layout_path)
    print("💾 Text saved:", text_path)

    return collected_text, text_regions


# ================= PDF RECONSTRUCTION =================
def reconstruct_pdf_with_visible_text(
    image_path,
    text_items,      # list of (y, x, text)
    text_boxes,      # list of (x1, y1, x2, y2)
    pdf_path
):
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

    # 1️⃣ Draw full image
    c.drawImage(
        ImageReader(image_path),
        0, 0,
        width=page_w,
        height=page_h
    )


    # 2️⃣ Mask text regions
    # (Not implemented in this function, but can be added if needed)

    # 5️⃣ Draw selectable OCR text (layout-aware)
    c.setFillColor(black)
    for y, x, text, label, box_h in text_items:
        pdf_x = x * scale_x
        pdf_y = page_h - (y * scale_y)

        # 🔑 Dynamic font size
        font_size = compute_font_size(box_h, scale_y)

        if label == "title":
            font_name = "Helvetica-Bold"
        elif label == "figure_caption":
            font_name = "Helvetica-Oblique"
        elif label == "table_caption":
            font_name = "Helvetica-Oblique"
        elif label == "table_footnote":
            font_name = "Helvetica-Oblique"
        elif label == "table":
            font_name = "Helvetica-Oblique"


        else:
            font_name = "Helvetica"

        c.setFont(font_name, font_size)
        c.drawString(pdf_x, pdf_y, text)

    c.save()




def pdf_to_images(pdf_path, dpi=300):
    """
    Convert PDF pages to images using PyMuPDF (NO poppler)
    """
    doc = fitz.open(pdf_path)

    image_paths = []
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



def process_pdf_to_pdf(input_pdf, output_pdf, temp_output_dir):
    """
    Full PDF → PDF pipeline using PyMuPDF
    """
    os.makedirs(temp_output_dir, exist_ok=True)

    # 1️⃣ Convert PDF to images
    page_images = pdf_to_images(input_pdf)

    c = canvas.Canvas(output_pdf, pagesize=A4)


    for page_idx, img_path in enumerate(page_images):
        print(f"📄 Processing page {page_idx + 1}")

        # 2️⃣ Run existing IMAGE pipeline
        collected_text, text_regions = draw_layout_boxes_and_ocr(
            img_path,
            temp_output_dir,
            page_num=page_idx + 1
        )

        img = cv2.imread(img_path)
        img_h, img_w = img.shape[:2]

        page_w, page_h = A4
        scale_x = page_w / img_w
        scale_y = page_h / img_h

        # 3️⃣ Draw page image
        c.drawImage(
            ImageReader(img_path),
            0, 0,
            width=page_w,
            height=page_h
        )

        # 4️⃣ Mask text regions
        c.setFillColor(white)
        for x1, y1, x2, y2 in text_regions:
            pdf_x = x1 * scale_x
            pdf_y = page_h - (y2 * scale_y)
            pdf_w = (x2 - x1) * scale_x
            pdf_h = (y2 - y1) * scale_y
            c.rect(pdf_x, pdf_y, pdf_w, pdf_h, fill=1, stroke=0)

        # 5️⃣ Draw selectable OCR text (layout-aware)
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




INPUT_PDF = r"C:\Users\sahas\Downloads\dynamic font sizing\merged.pdf"
OUTPUT_PDF = r"C:\Users\sahas\Downloads\ocr_readable\final_selectable.pdf"
TEMP_DIR = r"C:\Users\sahas\Downloads\ocr_readable\temp_output"

process_pdf_to_pdf(
    input_pdf=INPUT_PDF,
    output_pdf=OUTPUT_PDF,
    temp_output_dir=TEMP_DIR
)


