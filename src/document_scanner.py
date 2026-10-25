
import cv2
import numpy as np

## Grayscale → Gaussian Blur → Otsu → Morphological Closing → Contours → Four Corners

def detect_document(image):
    """
    Detect a document using Otsu thresholding
    and contour approximation.

    Returns:
        Four corner points or None.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    _, binary = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (7, 7)
    )

    closed = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel
    )

    contours, _ = cv2.findContours(
        closed,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    contours = sorted(
        contours,
        key=cv2.contourArea,
        reverse=True
    )

    image_area = image.shape[0] * image.shape[1]

    for contour in contours:

        if cv2.contourArea(contour) < 0.15 * image_area:
            continue

        perimeter = cv2.arcLength(contour, True)

        approx = cv2.approxPolyDP(
            contour,
            0.02 * perimeter,
            True
        )

        if len(approx) == 4 and cv2.isContourConvex(approx):
            return approx.reshape(4, 2)

    return None

## Top-Left → Top-Right → Bottom-Right → Bottom-Left


def order_points(pts):
    """
    Order four corner points:
    top-left, top-right, bottom-right, bottom-left.
    """
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)

    # Sort by vertical position
    y_sorted = pts[np.argsort(pts[:, 1])]

    # Sort top and bottom pairs horizontally
    top = y_sorted[:2]
    bottom = y_sorted[2:]

    tl, tr = top[np.argsort(top[:, 0])]
    bl, br = bottom[np.argsort(bottom[:, 0])]

    return np.array([tl, tr, br, bl], dtype=np.float32)



def perspective_transform(image, corners):
    """
    Apply a four-point perspective transformation.

    Args:
        image: Original BGR document image.
        corners: Four detected document corners.

    Returns:
        Perspective-corrected document image.
    """
    rect = order_points(corners)

    tl, tr, br, bl = rect

    # Calculate document dimensions
    width_bottom = np.linalg.norm(br - bl)
    width_top = np.linalg.norm(tr - tl)

    height_right = np.linalg.norm(tr - br)
    height_left = np.linalg.norm(tl - bl)

    max_width = max(2, int(round(max(width_bottom, width_top))))
    max_height = max(2, int(round(max(height_right, height_left))))

    # Target rectangle
    destination = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype=np.float32)

    # Calculate perspective matrix
    matrix = cv2.getPerspectiveTransform(
        rect, destination
    )

    # Apply transformation
    warped = cv2.warpPerspective(
        image,
        matrix,
        (max_width, max_height)
    )

    return warped
