
import cv2

from src.ocr_engine import OCREngine

# Load FUNSD document
image = cv2.imread(
    "dataset/training_data/images/0000971160.png"
)

if image is None:
    raise FileNotFoundError("FUNSD document not found.")

# Initialize OCR
engine = OCREngine(language="en", device="cpu")

# Extract text
results = engine.extract_text(image)

# Display results
for item in results:
    print(
        f"{item['text']} | "
        f"Confidence: {item['confidence']:.4f}"
    )

print("\nTotal Recognized Texts:", len(results))
