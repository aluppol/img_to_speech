from functools import singledispatchmethod
from enum import Enum, auto

class Label(Enum):
    HEADER = auto()
    FOOTER = auto()
    ANNOTATION = auto()
    TITLE = auto()
    CHAPTER_TITLE = auto()
    CHAPTER_TEXT = auto()
    OTHER = auto()
    ANNOTATION_REFERENCE = auto()
    EPIGRAPH = auto()
    AUTHOR = auto()

class LabelTransformer:
    def __init__(self):
        # Create mappings
        self.label_to_int = {label: label.value for label in Label}
        self.int_to_label = {label.value: label for label in Label}
        self.str_to_label = {label.name: label for label in Label}

    @singledispatchmethod
    def to_int(self, label_or_str):
        """Convert label enum or string to integer."""
        raise ValueError(f"Input must be a Label enum or a string. Got {type(label_or_str)}.")

    @to_int.register(Label)
    def _(self, label: Label):
        """Handle Label enum."""
        return self.label_to_int[label]

    @to_int.register(str)
    def _(self, label_str: str):
        """Handle string input."""
        if label_str not in self.str_to_label:
            raise ValueError(f"String '{label_str}' not recognized as a label.")
        label_enum = self.str_to_label[label_str]
        return self.label_to_int[label_enum]
        

    def to_str(self, index: int) -> Label:
        """Convert integer back to label enum."""
        if index not in self.int_to_label:
            raise ValueError(f"Index '{index}' not recognized.")
        return self.int_to_label[index]