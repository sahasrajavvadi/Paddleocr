"""
Utility functions for working with layout boxes and text formatting.
"""


def expand_box(x1, y1, x2, y2, img_w, img_h, pad_ratio=0.05, min_pad=10):
    """
    Expand box by a percentage + minimum pixels.
    Prevents top/bottom text from getting cut.
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
        inner[0] >= outer[0] - margin
        and inner[1] >= outer[1] - margin
        and inner[2] <= outer[2] + margin
        and inner[3] <= outer[3] + margin
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
            b.conf,
        ),
        reverse=True,
    )

    kept = []

    for box in boxes:
        discard = False
        for kept_box in kept:
            if box.cls != kept_box.cls:
                continue

            # 1) If box is fully inside a bigger box → DROP
            if is_inside(box.xyxy, kept_box.xyxy):
                discard = True
                break

            # 2) If heavy overlap → DROP
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
        reverse=True,
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


def compute_font_size(box_height_px, scale_y, min_size=6, max_size=18):
    """
    Compute a reasonable font size for the reconstructed PDF,
    based on the detected box height.
    """
    font_size = box_height_px * scale_y * 0.75
    return max(min_size, min(font_size, max_size))