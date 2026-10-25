
import cv2

from src.document_scanner import detect_document, perspective_transform
from src.image_preprocessing import to_grayscale, apply_clahe
from src.ocr_engine import OCREngine

# Load test document
image = cv2.imread(
    "dataset/training_data/images/0000971160.png"
)

if image is None:
    raise FileNotFoundError("Document not found.")

# Step 1: Detect and correct perspective
corners = detect_document(image)

if corners is not None:
    scanned = perspective_transform(image, corners)
    print("Perspective correction applied.")
else:
    scanned = image.copy()
    print("No document boundary found. Using original image.")

# Step 2: Image preprocessing
gray = to_grayscale(scanned)
enhanced = apply_clahe(gray)

# Step 3: OCR
engine = OCREngine(language="en", device="cpu")

results = engine.extract_text(enhanced)

# Display extracted text
print("\n===== EXTRACTED TEXT =====\n")

for item in results:
    print(item["text"])

print("\nTotal recognized segments:", len(results))
