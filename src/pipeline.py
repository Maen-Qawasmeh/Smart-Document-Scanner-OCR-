
import cv2

from src.document_scanner import (
    detect_document,
    perspective_transform
)

from src.image_preprocessing import (
    to_grayscale,
    apply_clahe,
    apply_adaptive_threshold
)

from src.ocr_engine import OCREngine


class DocumentPipeline:
    """
    Complete document scanning and OCR pipeline.
    """

    def __init__(self, language="en", device="cpu"):
        self.ocr_engine = OCREngine(
            language=language,
            device=device
        )

    def process(self, image, auto_scan=False, preprocessing="original"):
        """
        Process a document image and extract text.

        Args:
            image: Input BGR image.
            auto_scan: Apply document detection and perspective correction.
            preprocessing: original, grayscale, clahe, or adaptive.

        Returns:
            Dictionary containing the processed image,
            extracted text, and scanning status.
        """

        if image is None:
            raise ValueError("Input image cannot be None.")

        # Step 1: Document scanning
        scanned = image.copy()
        scan_status = "skipped"

        if auto_scan:
            corners = detect_document(image)

            if corners is not None:
                scanned = perspective_transform(image, corners)
                scan_status = "applied"
            else:
                scan_status = "not_detected"

        # Step 2: Image preprocessing
        if preprocessing == "original":
            processed = scanned.copy()

        elif preprocessing == "grayscale":
            processed = to_grayscale(scanned)

        elif preprocessing == "clahe":
            processed = apply_clahe(scanned)

        elif preprocessing == "adaptive":
            enhanced = apply_clahe(scanned)
            processed = apply_adaptive_threshold(enhanced)

        else:
            raise ValueError(
                f"Unsupported preprocessing method: {preprocessing}"
            )

        # Step 3: OCR
        extracted_text = self.ocr_engine.extract_text(processed)

        return {
            "scanned_image": scanned,
            "processed_image": processed,
            "text_results": extracted_text,
            "scan_status": scan_status,
            "preprocessing": preprocessing
        }
