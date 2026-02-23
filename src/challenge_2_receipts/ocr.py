from pathlib import Path
import warnings
import torch
import easyocr

warnings.filterwarnings("ignore", message=".*pin_memory.*")

reader = easyocr.Reader(["en"], gpu=torch.backends.mps.is_available())


def run_ocr(image_path: Path, confidence_threshold: float = 0.2) -> dict:
    detections = reader.readtext(str(image_path))
    filtered = [
        (bbox, text, conf)
        for bbox, text, conf in detections
        if conf >= confidence_threshold
    ]
    return {
        "raw_text": "\n".join([text for _, text, _ in filtered]),
        "detections": detections,
        "confidences": [conf for _, _, conf in detections],
    }
