import re

from fuzzywuzzy import fuzz


def clean_and_split_text(text):
    """
    Cleans and splits the input text into blocks of words.

    Args:
        text (str): The raw text to be cleaned and split.

    Returns:
        list of str: A list of cleaned and lowercased text blocks.
    """

    def clean_text(text):
        ## COMMENT AV: It could be a separate function outside the main function.
        text = re.sub(r"\d", "", text)
        text = re.sub(r"[^a-zA-Z\s]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    blocks = text.split("\n")
    blocks = [block.strip() for block in blocks if block.strip()]
    clean_blocks = [clean_text(bk) for bk in blocks]
    clean_blocks = [c_bk.lower() for c_bk in clean_blocks if c_bk not in ["", " "]]
    ## COMMENT AV: Nice use of list comprehensions, but do not be afraid to use more descriptive variable names, it will make your code more readable.
    return clean_blocks


def find_bounding_boxes(detected_names, ocr_data):
    ## COMMENT AV: Remember: It is a good practice to use typing to specify the types of the input and output parameters.
    ## def find_bounding_boxes(detected_names: List[str], ocr_data: Dict) -> List[Dict]:

    ## Further improvement: You can use a class to store the bounding box information, instead of a dictionary.
    ## I like to use pydantic for everything, but you can use dataclasses as well.
    ## It will get rid of the magic strings and will have better type checking.

    ## from pydantic import BaseModel
    ## class BoundingBox(BaseModel):
    ##     name: str
    ##     left: int
    ##     top: int
    ##     width: int
    ##     height: int

    """
    Finds bounding boxes for detected names in OCR data.

    Args:
        detected_names (list of str): A list of detected names to find in the OCR data.
        ocr_data (dict): OCR data containing text and bounding box information.

    Returns:
        list of dict: A list of bounding boxes for each detected name.
    """
    bounding_boxes = []
    for name in detected_names:
        for i, word in enumerate(ocr_data["text"]):
            if name.lower() in word.lower() and int(ocr_data["conf"][i]) > 60:
                bbox = {
                    "name": name,
                    "left": ocr_data["left"][i],
                    "top": ocr_data["top"][i],
                    "width": ocr_data["width"][i],
                    "height": ocr_data["height"][i],
                }
                bounding_boxes.append(bbox)
    return bounding_boxes


# drawing_bounding_boxes(processed_img.copy(),bounding_boxes)


def fuzzy_match_names(extracted_name, provided_name, threshold=90):
    """
    Performs fuzzy matching between an extracted name and a provided name.

    Args:
        extracted_name (str): The name extracted from the image.
        provided_name (str): The name provided by the user for comparison.
        threshold (int, optional): The similarity threshold for a match. Defaults to 90.

    Returns:
        tuple: A tuple containing the fuzzy similarity score (int) and a boolean indicating if the similarity is above the threshold.
    """
    fuzzy_similarity = fuzz.ratio(extracted_name, provided_name)
    similar_name = fuzzy_similarity >= threshold
    return fuzzy_similarity, similar_name
