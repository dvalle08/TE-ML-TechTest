import cv2
import numpy as np
import pytesseract


def extract_text_from_image(image):
    ## COMMENT AV: Same here for typing, you can specify the types of the input and output parameters.
    ## def extract_text_from_image(image: np.ndarray) -> Tuple[str, np.ndarray]:

    """
    Extracts text from an image after rotating and cropping it to focus on the document area.

    Args:
        image (numpy.ndarray): The input image from which text needs to be extracted.

    Returns:
        tuple: A tuple containing the extracted text (str) and the cropped image (numpy.ndarray).
    """
    rotated_image = rotate_image(image)
    crop_img = cropping_image(rotated_image)
    text = pytesseract.image_to_string(crop_img)
    return text, crop_img


def rotate_image(image):
    """
    Rotates the input image to align the document within it.

    Args:
        image (numpy.ndarray): The input image to be rotated.

    Returns:
        numpy.ndarray: The rotated image.
    """
    rect = get_document_contour(image)
    # ?show_image_with_bounding_box(rect, image.copy())
    angle = rect[-1]
    (width, height) = rect[1]
    if width < height:
        angle = angle - 90
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return rotated


def cropping_image(rotated_image):
    """
    Crops the rotated image to the bounding box of the document.

    Args:
        rotated_image (numpy.ndarray): The rotated image to be cropped.

    Returns:
        numpy.ndarray: The cropped image.
    """
    rect = get_document_contour(rotated_image)
    # ?show_image_with_bounding_box(rect, image.copy())
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    x, y, w, h = cv2.boundingRect(box)
    crop_img = rotated_image[y : y + h, x : x + w]
    return crop_img


def get_document_contour(image):
    """
    Detects and returns the minimum area rectangle around the largest contour in the image.

    Args:
        image (numpy.ndarray): The input image in which to detect the document contour.

    Returns:
        tuple: A tuple containing the center (x, y), size (width, height), and angle of rotation of the rectangle.
    """
    ## COMMENT AV: I tried this, it works, but depends a lot on the quality of the edges of the document, but it is a good approach.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 40))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest_contour)
    return rect
