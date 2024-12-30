import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
from PIL import UnidentifiedImageError
from src.Text_Extractor_Outdated import TextExtractor


class TestTextExtractor(unittest.TestCase):

    @patch("src.TextExtractor.Path.exists", return_value=True)
    @patch("src.TextExtractor.Path.is_file", return_value=True)
    def test_validate_file_path(self, mock_is_file, mock_exists):
      path = TextExtractor._TextExtractor__validate_file_path("test.jpg", {".jpg", ".png"})
      self.assertIsInstance(path, Path)
      self.assertEqual(path.name, "test.jpg")

    @patch("src.TextExtractor.Path.exists", return_value=False)
    def test_validate_file_path_not_found(self, mock_exists):
      with self.assertRaises(FileNotFoundError):
        TextExtractor._TextExtractor__validate_file_path("test.jpg", {".jpg", ".png"})

    @patch("src.TextExtractor.Path.exists", return_value=True)
    @patch("src.TextExtractor.Path.is_file", return_value=False)
    def test_validate_file_path_is_directory(self, mock_is_file, mock_exists):
      with self.assertRaises(IsADirectoryError):
        TextExtractor._TextExtractor__validate_file_path("test.jpg", {".jpg", ".png"})

    @patch("src.TextExtractor.Path.exists", return_value=True)
    @patch("src.TextExtractor.Path.is_file", return_value=True)
    def test_validate_file_path_invalid_extension(self, mock_is_file, mock_exists):
      with self.assertRaises(ValueError):
        TextExtractor._TextExtractor__validate_file_path("test.txt", {".jpg", ".png"})


    @patch("src.TextExtractor.TextExtractor._TextExtractor__pdf_to_text", return_value="PDF content")
    @patch("src.TextExtractor.TextExtractor._TextExtractor__validate_file_path")
    def test_extract_pdf(self, mock_validate, mock_pdf_to_text):
      mock_validate.return_value = Path("test.pdf")
      result = TextExtractor.extract("test.pdf")
      self.assertEqual(result, "PDF content")

    @patch("src.TextExtractor.Image.open", return_value=MagicMock())
    @patch("src.TextExtractor.pytesseract.image_to_string", return_value="Image content")
    @patch("src.TextExtractor.TextExtractor._TextExtractor__validate_file_path")
    def test_extract_from_image(self, mock_validate, mock_image_to_string, mock_open):
      mock_validate.return_value = Path("test.jpg")
      result = TextExtractor.extract("test.jpg")
      self.assertEqual(result, "Image content")

    @patch("src.TextExtractor.TextExtractor._TextExtractor__validate_file_path")
    def test_extract_from_unsupported_extension(self, mock_validate):
      mock_validate.return_value = Path("test.unsupported")
      with self.assertRaises(ValueError):
        TextExtractor.extract("test.unsupported")

    @patch("src.TextExtractor.pymupdf")
    def test_pdf_to_text_valid(self, mock_pymupdf):
      # Mock the document and pages
      mock_doc = MagicMock()
      mock_page = MagicMock()
      mock_page.number = 0

      mock_block = {
        "lines": [
          {
            "spans": [
              {"text": "Chapter 1: Introduction", "size": 18, "flags": 2, "bbox": (0, 0, 100, 50)},
              {"text": "This is a sample text.", "size": 12, "flags": 0, "bbox": (0, 60, 100, 100)}
            ]
          }
        ]
      }
      mock_page.get_text.return_value = {"blocks": [mock_block]}
      mock_doc.__iter__.return_value = [mock_page]  # Mock page access
      mock_doc.__len__.return_value = 1  # Single page document
      mock_pymupdf.open.return_value = mock_doc

      # Run the method
      result = TextExtractor._TextExtractor__pdf_to_text("test.pdf")

      # Expected result
      expected = [
        {"text": "Chapter 1: Introduction", "size": 18, "flags": 2, "bbox": (0, 0, 100, 50), "page": 1},
        {"text": "This is a sample text.", "size": 12, "flags": 0, "bbox": (0, 60, 100, 100), "page": 1}
      ]

      # Assert
      self.assertEqual(result, expected)

    @patch("src.TextExtractor.pymupdf")
    def test_pdf_to_text_empty_pdf(self, mock_pymupdf):
      # Mock an empty document
      mock_doc = MagicMock()
      mock_doc.__len__.return_value = 0
      mock_pymupdf.open.return_value = mock_doc

      # Run the method
      result = TextExtractor._TextExtractor__pdf_to_text("empty.pdf")

      # Assert the result is an empty list
      self.assertEqual(result, [])

    @patch("src.TextExtractor.pymupdf")
    def test_pdf_to_text_multiple_pages(self, mock_pymupdf):
      # Mock the document and multiple pages
      mock_doc = MagicMock()
      mock_page1 = MagicMock()
      mock_page1.number = 0
      mock_page2 = MagicMock()
      mock_page2.number = 1

      # Mock text metadata for two pages
      mock_page1.get_text.return_value = {
          "blocks": [
              {"lines": [{"spans": [{"text": "Page 1: Title", "size": 16, "flags": 2, "bbox": (0, 0, 100, 50)}]}]}
          ]
      }
      mock_page2.get_text.return_value = {
          "blocks": [
              {"lines": [{"spans": [{"text": "Page 2: Content", "size": 12, "flags": 0, "bbox": (0, 60, 100, 100)}]}]}
          ]
      }

      mock_doc.__iter__.return_value = [mock_page1, mock_page2]  # Mock page access
      mock_doc.__len__.return_value = 2  # Two pages
      mock_pymupdf.open.return_value = mock_doc

      # Run the method
      result = TextExtractor._TextExtractor__pdf_to_text("empty.pdf")

      # Expected result
      expected = [
        {"text": "Page 1: Title", "size": 16, "flags": 2, "bbox": (0, 0, 100, 50), "page": 1},
        {"text": "Page 2: Content", "size": 12, "flags": 0, "bbox": (0, 60, 100, 100), "page": 2}
      ]

      # Assert
      self.assertEqual(result, expected)


    @patch("src.TextExtractor.Image.open")
    @patch("src.TextExtractor.pytesseract.image_to_string", return_value="Image content")
    def test_picture_to_text_valid(self, mock_image_to_string, mock_open):
      result = TextExtractor._TextExtractor__picture_to_text(Path("test.jpg"))
      self.assertEqual(result, "Image content")

    @patch("src.TextExtractor.Image.open", side_effect=UnidentifiedImageError)
    def test_picture_to_text_invalid_image(self, mock_open):
      with self.assertRaises(ValueError):
        TextExtractor._TextExtractor__picture_to_text(Path("test.jpg"))

    @patch("src.TextExtractor.Image.open", side_effect=Exception("Unexpected error"))
    def test_picture_to_text_runtime_error(self, mock_open):
      with self.assertRaises(RuntimeError):
        TextExtractor._TextExtractor__picture_to_text(Path("test.jpg"))


if __name__ == "__main__":
    unittest.main()















