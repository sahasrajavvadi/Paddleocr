import abc
import ast
import os
from typing import List

import cv2
import numpy as np

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


class DocLayoutModel(abc.ABC):
    """
    Abstract base class for document layout models.
    Currently only ONNX implementation is provided.
    """

    @staticmethod
    def load_onnx() -> "OnnxModel":
        """
        Load the default ONNX layout model.
        """
        model = OnnxModel.from_pretrained()
        return model

    @staticmethod
    def load_available() -> "OnnxModel":
        """
        Factory method – if in future you add more backends,
        you can decide which one to load here.
        """
        return DocLayoutModel.load_onnx()

    @property
    @abc.abstractmethod
    def stride(self) -> int:
        pass

    @abc.abstractmethod
    def predict(self, image, imgsz: int = 1024, **kwargs) -> list:
        pass


class YoloBox:
    """
    Simple container for a YOLO box prediction.
    """

    def __init__(self, data):
        self.xyxy = data[:4]
        self.conf = data[-2]
        self.cls = data[-1]


class YoloResult:
    """
    Container for YOLO prediction results.
    """

    def __init__(self, boxes, names):
        self.boxes = [YoloBox(data=d) for d in boxes]
        # sort by confidence (high → low)
        self.boxes.sort(key=lambda x: x.conf, reverse=True)
        self.names = names


class OnnxModel(DocLayoutModel):
    """
    ONNX implementation of the document layout model.
    """

    # NOTE: change this path if you move the model
    DEFAULT_MODEL_PATH = (
        r"C:\Users\sahas\Downloads\pdfocr\doclayout_yolo_docstructbench_imgsz1024.onnx"
    )

    def __init__(self, model_path: str):
        self.model_path = model_path

        model = onnx.load(model_path)
        metadata = {d.key: d.value for d in model.metadata_props}
        self._stride = ast.literal_eval(metadata["stride"])
        self._names = ast.literal_eval(metadata["names"])

        self.model = onnxruntime.InferenceSession(
            model.SerializeToString(),
            providers=["CPUExecutionProvider"],
        )

    @staticmethod
    def from_pretrained() -> "OnnxModel":
        """
        Load ONNX model from the default path.
        """
        pth = OnnxModel.DEFAULT_MODEL_PATH
        if not os.path.exists(pth):
            raise FileNotFoundError(f"ONNX model not found at: {pth}")
        return OnnxModel(pth)

    @property
    def stride(self) -> int:
        return self._stride

    @property
    def names(self) -> List[str]:
        return self._names

    # ------------- internal helpers -------------
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
            image,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )
        return image

    @staticmethod
    def scale_boxes(img1_shape, boxes, img0_shape):
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad_x = round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1)
        pad_y = round((img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1)

        boxes[..., :4] = (boxes[..., :4] - [pad_x, pad_y, pad_x, pad_y]) / gain
        return boxes

    # ------------- main inference -------------
    def predict(self, image, imgsz: int = 1024, **kwargs) -> list:
        """
        Run layout detection on a BGR image.
        Returns a list with a single YoloResult.
        """
        orig_h, orig_w = image.shape[:2]
        pix = self.resize_and_pad_image(image, imgsz)
        pix = np.transpose(pix, (2, 0, 1))
        pix = np.expand_dims(pix, axis=0)
        pix = pix.astype(np.float32) / 255.0

        new_h, new_w = pix.shape[2:]
        preds = self.model.run(None, {"images": pix})[0]

        # IMPORTANT: lower threshold (from original code)
        preds = preds[preds[..., 4] > 0.10]
        preds[..., :4] = self.scale_boxes(
            (new_h, new_w),
            preds[..., :4],
            (orig_h, orig_w),
        )

        return [YoloResult(boxes=preds, names=self._names)]