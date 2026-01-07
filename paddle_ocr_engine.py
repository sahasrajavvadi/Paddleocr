from typing import Optional
from paddleocr import PaddleOCR

# Singleton instance
_ocr_instance: Optional[PaddleOCR] = None


def get_paddle_ocr() -> PaddleOCR:
    global _ocr_instance

    if _ocr_instance is not None:
        return _ocr_instance

    _ocr_instance = PaddleOCR(
        lang='en',
        use_angle_cls=True,
        det_model_dir=r"C:\Users\sahas\Downloads\ocr_readable\paddleocrmodels\det",
        rec_model_dir=r"C:\Users\sahas\Downloads\ocr_readable\paddleocrmodels\rec",
        cls_model_dir=r"C:\Users\sahas\Downloads\ocr_readable\paddleocrmodels\cls",
        use_gpu=False
    )

    print("✅ PaddleOCR ready (OFFLINE)")
    return _ocr_instance


if __name__ == "__main__":
    ocr = get_paddle_ocr()
    print("Detection model:", ocr.det_model_dir)
    print("Recognition model:", ocr.rec_model_dir)
    print("CLS model:", ocr.cls_model_dir)
