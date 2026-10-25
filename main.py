print("MAIN.PY IS RUNNING!")
import cv2
from pathlib import Path

from src.document_scanner import (
    detect_document,
    perspective_transform
)

# Load document
image_path = Path("sample_images/document_tilted.jpg")
image = cv2.imread(str(image_path))

if image is None:
    raise FileNotFoundError(image_path)

# Detect four corners
corners = detect_document(image)

if corners is None:
    print("Document not detected.")

else:
    print("Detected corners:")
    print(corners)

    # Correct perspective
    scanned = perspective_transform(image, corners)

    # Save output
    output_path = Path("outputs/scanned_document.jpg")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(output_path), scanned)

    print("Scanned document saved:", output_path)
    print("Output shape:", scanned.shape)


    #image preprocessing
    from src.image_preprocessing import (
        to_grayscale,
        apply_clahe,
        apply_adaptive_threshold
    )

    # Generate image variants
    gray = to_grayscale(scanned)
    enhanced = apply_clahe(scanned)
    binary = apply_adaptive_threshold(enhanced)

    # Save results
    cv2.imwrite("outputs/scanned_gray.jpg", gray)
    cv2.imwrite("outputs/scanned_clahe.jpg", enhanced)
    cv2.imwrite("outputs/scanned_binary.png", binary)

    print("Image preprocessing completed successfully!")

