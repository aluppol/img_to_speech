import pytesseract
from PIL import Image, UnidentifiedImageError
from pathlib import Path
from typing import Optional, Set, TypedDict, Tuple, List, Generator
import pymupdf
from pdf2image import convert_from_path
from abc import ABC, abstractmethod
import cv2
import numpy as np


class FeaturedText(TypedDict):
    text: str  # The text content (e.g., a word or phrase).
    font_size: float  # The font size or relative size of the text.
    style_flags: int  # Flags providing metadata (e.g., styling or annotations).
    bbox: Tuple[float, float, float, float]  # Bounding box (x1, y1, x2, y2).


class TextExtractor(ABC):
    """An abstract class to handle text extraction."""
    
    @staticmethod
    def _validate_file_path(file_path: str, expected_extensions: Optional[Set[str]] = None) -> Path:
        """
        Validate the file path and extension.

        Args:
            file_path (str): Path to the file to validate.
            expected_extensions (set): A set of supported file extensions (e.g., {".jpg", ".pdf"}).

        Raises:
            FileNotFoundError: If the file does not exist.
            IsADirectoryError: If the provided path is a directory instead of a file.
            ValueError: If the file extension is not in the expected set.

        Returns:
            pathlib.Path: The validated file path as a Path object.
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if not path.is_file():
            raise IsADirectoryError(f"Path is not a file: {file_path}")
        if expected_extensions and path.suffix.lower() not in expected_extensions:
            raise ValueError(f'Unsupported file extention: "{path.suffix}". Expected extensions: {expected_extensions}')
        return path

    @abstractmethod
    def extract(file_path: str, from_page: int = None, to_page: int = None) -> List[FeaturedText]:
        pass

class PdfTextExtractor(TextExtractor):
    def extract(self, file_path: str, from_page: int = None, to_page: int = None) -> List[FeaturedText]:
        """
        Raises:
            FileNotFoundError: If the file does not exist.
            IsADirectoryError: If the provided path is a directory instead of a file.
            ValueError: If the file extension is not in the expected set.
        """
        pdf_path = self._validate_file_path(file_path=file_path, expected_extensions=set(['.pdf']))
        pages = convert_from_path(pdf_path, dpi=300)
        for page in pages:
            opencv_format = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
            grayscale = cv2.cvtColor(opencv_format, cv2.COLOR_RGB2BGR)
            _, thresh = cv2.threshold(grayscale, 150, 255, cv2.THRESH_BINARY)
            text = pytesseract.image_to_data(thresh, output_type=pytesseract.Output.DICT)