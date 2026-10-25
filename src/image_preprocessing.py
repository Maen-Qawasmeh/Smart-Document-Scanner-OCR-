
import cv2


def to_grayscale(image):
    """
    Convert a BGR image to grayscale.
    """
    if image.ndim == 2:
        return image.copy()

    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def apply_clahe(image, clip_limit=2.0, tile_size=(8, 8)):
    """
    Enhance local contrast using CLAHE.
    """
    gray = to_grayscale(image)

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_size
    )

    return clahe.apply(gray)


def apply_adaptive_threshold(image, block_size=31, c=10):
    """
    Apply adaptive Gaussian thresholding.
    """
    gray = to_grayscale(image)

    if block_size <= 1 or block_size % 2 == 0:
        raise ValueError(
            "block_size must be an odd integer greater than 1."
        )

    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        c
    )

    return binary
