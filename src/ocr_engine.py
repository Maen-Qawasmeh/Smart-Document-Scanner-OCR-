
import cv2
from paddleocr import PaddleOCR


class OCREngine:
    """
    OCR engine for extracting text and confidence
    scores from document images using PaddleOCR.
    """

    def __init__(self, language="en", device="cpu"):
        """
        Initialize PaddleOCR using lightweight models
        suitable for CPU and cloud deployment.
        """

        print(
            "Starting PaddleOCR initialization...",
            flush=True
        )

        self.ocr = PaddleOCR(
            lang=language,
            device=device,

            # Lightweight OCR models
            text_detection_model_name="PP-OCRv5_mobile_det",
            text_recognition_model_name="en_PP-OCRv5_mobile_rec",

            # Disable optional document processing
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,

            # CPU optimization
            enable_mkldnn=False,
            cpu_threads=2
        )

        print(
            "PaddleOCR initialized successfully!",
            flush=True
        )

    def extract_text(self, image):
        """
        Extract recognized text and confidence scores
        from BGR or grayscale document images.

        Args:
            image: Input document image (NumPy array).

        Returns:
            list: Dictionaries containing recognized text
                  and confidence scores.
        """

        # Validate input image
        if image is None:
            raise ValueError(
                "Input image cannot be None."
            )

        # Convert grayscale images to BGR
        if image.ndim == 2:
            image = cv2.cvtColor(
                image,
                cv2.COLOR_GRAY2BGR
            )

        # Run OCR prediction
        results = self.ocr.predict(image)

        extracted = []

        # Extract recognized texts and confidence scores
        for result in results:

            texts = result["rec_texts"]
            scores = result["rec_scores"]

            for text, score in zip(texts, scores):

                extracted.append({
                    "text": text,
                    "confidence": float(score)
                })

        return extracted
