from src.export_utils import export_ocr_results
import cv2

from src.pipeline import DocumentPipeline

# Load a document containing text
image = cv2.imread(
    "dataset/training_data/images/0000971160.png"
)

if image is None:
    raise FileNotFoundError("Test document not found.")

# Initialize pipeline
pipeline = DocumentPipeline(
    language="en",
    device="cpu"
)

# Process document
result = pipeline.process(
    image,
    auto_scan=False,
    preprocessing="original"
)

print("\n===== PIPELINE RESULTS =====")

print("Scan Status:", result["scan_status"])
print("Preprocessing:", result["preprocessing"])
print("Recognized Texts:", len(result["text_results"]))

print("\n===== EXTRACTED TEXT =====\n")

for item in result["text_results"]:
    print(item["text"])


# Export OCR results
exported_files = export_ocr_results(
    result,
    output_dir="outputs",
    filename="funsd_test"
)

print("\n===== EXPORTED FILES =====")
print("TXT:", exported_files["txt_path"])
print("JSON:", exported_files["json_path"])
