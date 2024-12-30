import pytesseract
from PIL import Image
from pathlib import Path
from typing import Optional, Set, Dict, Any, List, Generator
from pdf2image import convert_from_path
from abc import ABC, abstractmethod
import cv2
import numpy as np


class FeaturedText():
    def __init__(
            self,
            level: int,
            page_number: int,
            block_number: int,
            paragraph_number: int, 
            line_number: int,
            word_number: int,
            left_position: int,
            top_position: int, 
            box_width: int,
            box_height: int,
            confidence_score: int,
            text_content: str,
        ):
        self.level: int = level
        self.page_number: int = page_number
        self.block_number: int = block_number
        self.paragraph_number: int = paragraph_number
        self.line_number: int = line_number
        self.word_number: int = word_number
        self.left_position: int = left_position
        self.top_position: int = top_position
        self.box_width: int = box_width
        self.box_height: int = box_height
        self.confidence_score: int = confidence_score
        self.text_content: str = text_content

    @classmethod
    def from_recognized_data(cls, recognized_image_data: Dict[str, Any], index: int) -> 'FeaturedText':
        return cls(
            level=recognized_image_data['level'][index],
            page_number=recognized_image_data['page_num'][index],
            block_number=recognized_image_data['block_num'][index],
            paragraph_number=recognized_image_data['par_num'][index],
            line_number=recognized_image_data['line_num'][index],
            word_number=recognized_image_data['word_num'][index],
            left_position=recognized_image_data['left'][index],
            top_position=recognized_image_data['top'][index],
            box_width=recognized_image_data['width'][index],
            box_height=recognized_image_data['height'][index],
            confidence_score=recognized_image_data['conf'][index],
            text_content=recognized_image_data['text'][index]
        )
    
    def __str__(self) -> str:
        return (
            f"Text: {self.text_content}\n"
            f"Level: {self.level}\n"
            f"Page Number: {self.page_number}\n"
            f"Block Number: {self.block_number}\n"
            f"Paragraph Number: {self.paragraph_number}\n"
            f"Line Number: {self.line_number}\n"
            f"Word Number: {self.word_number}\n"
            f"Bounding Box - Left: {self.left_position}, Top: {self.top_position}, Width: {self.box_width}, Height: {self.box_height}\n"
            f"Confidence Score: {self.confidence_score}"
        )


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
    def extract(file_path: str) -> List[FeaturedText]:
        pass


class PdfTextExtractor(TextExtractor):
    def extract(self, pdf_file_path: str) -> Generator[FeaturedText, None, None]:
        for page_image in self.__extract_page_images_from_pdf_path(pdf_file_path):
            for featured_text in self.__extract_featured_text_from_image(page_image):
                yield featured_text

    def __extract_page_images_from_pdf_path(self, pdf_file_path: str) -> List[Image.Image]:
        pdf_path = self._validate_file_path(file_path=pdf_file_path, expected_extensions=set(['.pdf']))
        return convert_from_path(pdf_path, dpi=300)

    def __extract_featured_text_from_image(self, image: Image) -> List[FeaturedText]:
        prepared_image = self.__prepare_image_for_text_extraction(image=image)
        return self.__extract_featured_text_from_prepared_image(image=prepared_image)

    @staticmethod
    def __prepare_image_for_text_extraction(image: Image) -> cv2.typing.MatLike:
        opencv_format = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        grayscale = cv2.cvtColor(opencv_format, cv2.COLOR_RGB2BGR)
        _, thresh = cv2.threshold(grayscale, 150, 255, cv2.THRESH_BINARY)
        return thresh
    
    @staticmethod
    def __extract_featured_text_from_prepared_image(image: cv2.typing.MatLike) -> List[FeaturedText]:
        result: List[FeaturedText] = []
        recognized_image_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        for i in range(len(next(iter(recognized_image_data.values())))):
            result.append(FeaturedText.from_recognized_data(recognized_image_data=recognized_image_data, index=i))
        return result
            