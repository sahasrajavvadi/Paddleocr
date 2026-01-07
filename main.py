"""
Main entry-point for the OCR + PDF pipeline.

Your manager can just open this file to see:
- where the input PDF is
- where the output PDF will be written
- how to run the whole process
"""

from pdf_pipeline import process_pdf_to_pdf
from paddle_ocr_engine import get_paddle_ocr


def run_pipeline() -> None:
    """
    Configure your input/output paths here and run the full pipeline.
    """
    # TODO: update these three paths to match your environment
    input_pdf = r"C:\Users\sahas\Downloads\paddleocrfinal\2511.05667v1 (1).pdf"
    output_pdf = r"C:\Users\sahas\Downloads\ocr_readable\final_selectable.pdf"
    temp_dir = r"C:\Users\sahas\Downloads\ocr_readable\temp_output"

    # Make sure PaddleOCR can load before we start
    ocr = get_paddle_ocr()
    
    # Run the PDF → PDF pipeline
    process_pdf_to_pdf(
        input_pdf=input_pdf,
        output_pdf=output_pdf,
        temp_output_dir=temp_dir,
    )


if __name__ == "__main__":
    run_pipeline()