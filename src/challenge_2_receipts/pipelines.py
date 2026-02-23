import base64
import json
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from .schemas import ReceiptEntities
from .ocr import run_ocr

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
entity_llm = llm.with_structured_output(ReceiptEntities)


def encode_image(image_path: Path) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def improved_ocr(image_path: Path) -> str:
    encoded = encode_image(image_path)
    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": "Extract all text you see in this receipt image exactly as it appears. Preserve the original content and structure.",
            },
            {"type": "image", "base64": encoded, "mime_type": "image/jpeg"},
        ]
    )

    result = llm.invoke([message]).content
    if isinstance(result, list):
        return " ".join(
            part["text"] if isinstance(part, dict) else str(part) for part in result
        )
    return result


def extract_entities(text: str) -> ReceiptEntities:
    prompt = f"""You are analyzing OCR text from a scanned receipt.
Extract the required entities and in doing so:

1. Rectify malformed words caused by OCR misreads:
   - Character confusion (e.g. '0' misread as 'O', '1' as 'l', 'rn' as 'm')
   - Broken words (e.g. 'To tal' should be 'Total')
   - Garbled text (e.g. 'sollor' should be 'seller')

2. Complete compound entities with missing components:
   - Use world knowledge to complete partial business names
     (e.g. 'MYDIN MO' should be completed to 'MYDIN MOHAMED HOLDINGS SDN BHD')
   - Complete partial addresses using surrounding context clues

3. Apply any other necessary corrections:
   - Normalize spacing and punctuation where clearly wrong
   - Fix split lines that belong together
   - Correct currency symbols or amount formatting if malformed

Important:
- Do not add information that cannot be inferred from the text or world knowledge
- For dates, extract the date only — do not include time or timezone

Receipt OCR text:
{text}"""

    return entity_llm.invoke([HumanMessage(content=prompt)])


def unified_pipeline(image_path: Path) -> ReceiptEntities:
    encoded = encode_image(image_path)
    prompt = """You are processing a scanned receipt image.
Extract the required entities directly from the image and in doing so:

1. Rectify malformed words caused by OCR misreads:
   - Character confusion (e.g. '0' misread as 'O', '1' as 'l', 'rn' as 'm')
   - Broken words (e.g. 'To tal' should be 'Total')
   - Garbled text (e.g. 'sollor' should be 'seller')

2. Complete compound entities with missing components:
   - Use world knowledge to complete partial business names
     (e.g. 'MYDIN MO' should be completed to 'MYDIN MOHAMED HOLDINGS SDN BHD')
   - Complete partial addresses using surrounding context clues

3. Apply any other necessary corrections:
   - Normalize spacing and punctuation where clearly wrong
   - Fix split lines that belong together
   - Correct currency symbols or amount formatting if malformed

Important:
- Do not add information that cannot be inferred from the image or world knowledge
- For dates, extract the date only — do not include time or timezone"""

    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {"type": "image", "base64": encoded, "mime_type": "image/jpeg"},
        ]
    )
    return entity_llm.invoke([message])


def run_pipeline(image_path: Path, tier: str) -> dict:
    if tier == "Tier 1":
        raw_text = run_ocr(image_path)["raw_text"]
        return {"context": raw_text, "display": raw_text}

    elif tier == "Tier 2":
        improved_text = improved_ocr(image_path)
        return {"context": improved_text, "display": improved_text}

    elif tier == "Tier 3":
        raw_text = run_ocr(image_path)["raw_text"]
        entities = extract_entities(raw_text).model_dump()
        return {"context": json.dumps(entities, indent=2), "display": entities}

    elif tier == "Tier 4":
        improved_text = improved_ocr(image_path)
        entities = extract_entities(improved_text).model_dump()
        return {"context": json.dumps(entities, indent=2), "display": entities}

    elif tier == "Tier 5":
        entities = unified_pipeline(image_path).model_dump()
        return {"context": json.dumps(entities, indent=2), "display": entities}
