import pytesseract
from PIL import Image, UnidentifiedImageError
from PyPDF2 import PdfReader
from pathlib import Path
from typing import Optional, Set



class ImgToTextConverter:
    """A class to handle text extraction from any type of img based files."""
    

    @staticmethod
    def __validate_file_path(file_path: str, expected_extensions: Optional[Set[str]] = None) -> Path:
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
            raise ValueError(f"Unsupported file type: {file_path}. Expected extensions: {expected_extensions}")
        return path


    @staticmethod
    def convert(image_path: str) -> str:
        """
        Master method to convert an image to text using Tesseract OCR.

        Args:
            image_path (str): Path to the image file.

        Raises:
            FileNotFoundError: If the file does not exist.
            IsADirectoryError: If the provided path is a directory instead of a file.
            ValueError: If the file extension is not supported or the image cannot be processed.
            RuntimeError: If any other error occurs during processing.

        Returns:
            str: Extracted text from the image.
        """
        try:
            # supported extensions
            supported_extensions = set()

            picture_extensions = set(Image.register_extensions())
            supported_extensions.update(picture_extensions)
            supported_extensions.add('pdf')

            path = ImgToTextConverter.__validate_file_path(image_path, supported_extensions)

            match path.suffix.lower():
                case 'pdf':
                    return ImgToTextConverter.__pdf_to_text(path)
                
                case _ if ext in picture_extensions:
                    return ImgToTextConverter.__picture_to_text(path)
                
                case _:
                    raise f"Provided extension is not implemented ${path.suffix.lower()}"

            
        except UnidentifiedImageError:
            raise ValueError(f"Cannot process the image: {image_path}")
        except Exception as e:
            raise RuntimeError(f"An error occurred while processing the image: {str(e)}")


    @staticmethod
    def __pdf_to_text(path: Path) -> str:
        """
        Convert a PDF to text.

        Args:
            pdf_path (str): Path to the PDF file.

        Raises:
            RuntimeError: If an error occurs during processing.

        Returns:
            str: Extracted text from the PDF.
        """
        try:
            reader = PdfReader(path)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
            return text if text else "No text found in PDF."
        except Exception as e:
            raise RuntimeError(f"An error occurred while processing the PDF: {str(e)}")


    @staticmethod
    def __picture_to_text(path: Path) -> str:
        """
        Convert an image to text using Tesseract OCR.

        Args:
            image_path (str): Path to the image file.

        Raises:
            ValueError: If the file cannot be processed as an image.
            RuntimeError: If an error occurs during processing.

        Returns:
            str: Extracted text from the image.
        """
        try:
            image = Image.open(path)
            return pytesseract.image_to_string(image)
        except UnidentifiedImageError:
            raise ValueError(f"Cannot process the image: {path}")
        except Exception as e:
            raise RuntimeError(f"An error occurred while processing the image: {str(e)}")