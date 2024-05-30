from transformers import pipeline


def person_name_entity_recognition(cleaned_text):
    ## COMMENT AV: Use typing to specify the types of the input and output parameters.
    ## This will help you and other developers understand the function's purpose and usage.
    ## It will also help IDEs provide better autocompletion and type checking.
    ## For example, cleaned_text: List[str] -> cleaned_text: List[str]
    ## COMMENT AV: Personal Preference: I would use a more descriptive name for the function, such as recognize_person_names.
    ## This makes the function's purpose clearer and easier to understand.
    ## I personally do not like docstrings for modules not intended to be used as libraries. They are unnecessary and add noise to the code.
    ## The function naming, variables and typing notations should be enough to understand the purpose of the function.
    ## def recognize_person_names_from_list_str(cleaned_text: List[str]) -> List[str]:
    ##
    """
    Recognizes person names in the cleaned text using a zero-shot classification model.

    Args:
        cleaned_text (list of str): A list of cleaned text strings to be analyzed for person names.

    Returns:
        list of str: A list of recognized person names in uppercase.
    """
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    labels = ["Person Name"]
    person_names = []
    for split_t in cleaned_text:
        results = classifier(split_t, candidate_labels=labels)
        if results["scores"][0] > 0.95:
            ## COMMENT AV: You can use a constant to define the threshold value, instead of a hardcoded value.
            ## Or include it as a parameter in the function signature.
            ## A more OOP approach would be to create a class with a method to set the threshold value.
            ## For example, class PersonNameRecognizer:
            ## Attribute: threshold = 0.95
            ## Attribute: classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
            ## Method: def recognize_person_names(self, cleaned_text: List[str]) -> List[str]:
            ## This way, you can easily change the threshold value or the model without modifying the function's code.
            ## I would suggest to set the threshold value as an environment variable so it can be easily changed without modifying the code. and avoid unnecessary commits to the repository when changing the threshold value.
            ## This is more Java like, but it is good to follow SOLID principles, especially the Dependency Injection principle and Liskov Substitution principle.
            ## At the long run, it will make your code more maintainable and scalable.
            person_names.append(results["sequence"].upper())
    return person_names
