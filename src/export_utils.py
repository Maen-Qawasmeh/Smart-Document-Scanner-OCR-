
import json
from pathlib import Path


def export_ocr_results(result, output_dir="outputs", filename="document"):
    """
    Export OCR pipeline results to TXT and JSON files.
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Extract OCR data
    texts = result["text_results"]

    # Save plain text
    txt_path = output_path / f"{filename}.txt"

    with open(txt_path, "w", encoding="utf-8") as file:
        for item in texts:
            file.write(item["text"] + "\n")

    # Prepare structured data
    json_data = {
        "scan_status": result["scan_status"],
        "preprocessing": result["preprocessing"],
        "total_text_segments": len(texts),
        "ocr_results": texts
    }

    # Save JSON
    json_path = output_path / f"{filename}.json"

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(
            json_data,
            file,
            indent=4,
            ensure_ascii=False
        )

    return {
        "txt_path": str(txt_path),
        "json_path": str(json_path)
    }
