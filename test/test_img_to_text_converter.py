import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
from PIL import UnidentifiedImageError
from src.img_to_text_converter import ImgToTextConverter


class TestImgToTextConverter(unittest.TestCase):

    @patch("src.img_to_text_converter.Path.exists", return_value=True)
    @patch("src.img_to_text_converter.Path.is_file", return_value=True)
    def test_validate_file_path(self, mock_is_file, mock_exists):
        path = ImgToTextConverter._ImgToTextConverter__validate_file_path("test.jpg", {".jpg", ".png"})
        self.assertIsInstance(path, Path)
        self.assertEqual(path.name, "test.jpg")

    @patch("src.img_to_text_converter.Path.exists", return_value=False)
    def test_validate_file_path_not_found(self, mock_exists):
        with self.assertRaises(FileNotFoundError):
            ImgToTextConverter._ImgToTextConverter__validate_file_path("test.jpg", {".jpg", ".png"})

    @patch("src.img_to_text_converter.Path.exists", return_value=True)
    @patch("src.img_to_text_converter.Path.is_file", return_value=False)
    def test_validate_file_path_is_directory(self, mock_is_file, mock_exists):
        with self.assertRaises(IsADirectoryError):
            ImgToTextConverter._ImgToTextConverter__validate_file_path("test.jpg", {".jpg", ".png"})

    @patch("src.img_to_text_converter.Path.exists", return_value=True)
    @patch("src.img_to_text_converter.Path.is_file", return_value=True)
    def test_validate_file_path_invalid_extension(self, mock_is_file, mock_exists):
        with self.assertRaises(ValueError):
            ImgToTextConverter._ImgToTextConverter__validate_file_path("test.txt", {".jpg", ".png"})


    @patch("src.img_to_text_converter.ImgToTextConverter._ImgToTextConverter__pdf_to_text", return_value="PDF content")
    @patch("src.img_to_text_converter.ImgToTextConverter._ImgToTextConverter__validate_file_path")
    def test_convert_pdf(self, mock_validate, mock_pdf_to_text):
        mock_validate.return_value = Path("test.pdf")
        result = ImgToTextConverter.convert("test.pdf")
        self.assertEqual(result, "PDF content")

    @patch("src.img_to_text_converter.Image.open", return_value=MagicMock())
    @patch("src.img_to_text_converter.pytesseract.image_to_string", return_value="Image content")
    @patch("src.img_to_text_converter.ImgToTextConverter._ImgToTextConverter__validate_file_path")
    def test_convert_image(self, mock_validate, mock_image_to_string, mock_open):
        mock_validate.return_value = Path("test.jpg")
        result = ImgToTextConverter.convert("test.jpg")
        self.assertEqual(result, "Image content")

    @patch("src.img_to_text_converter.ImgToTextConverter._ImgToTextConverter__validate_file_path")
    def test_convert_unsupported_extension(self, mock_validate):
        mock_validate.return_value = Path("test.unsupported")
        with self.assertRaises(ValueError):
            ImgToTextConverter.convert("test.unsupported")

    @patch("src.img_to_text_converter.PdfReader")
    def test_pdf_to_text_valid(self, mock_pdf_reader):
        mock_reader = MagicMock()
        mock_reader.pages = [MagicMock(extract_text=lambda: "Page 1"), MagicMock(extract_text=lambda: "Page 2")]
        mock_pdf_reader.return_value = mock_reader

        result = ImgToTextConverter._ImgToTextConverter__pdf_to_text(Path("test.pdf"))
        self.assertEqual(result, "Page 1Page 2")

    @patch("src.img_to_text_converter.PdfReader")
    def test_pdf_to_text_empty(self, mock_pdf_reader):
        mock_reader = MagicMock()
        mock_reader.pages = [MagicMock(extract_text=lambda: "")]
        mock_pdf_reader.return_value = mock_reader

        result = ImgToTextConverter._ImgToTextConverter__pdf_to_text(Path("test.pdf"))
        self.assertEqual(result, "No text found in PDF.")

    @patch("src.img_to_text_converter.Image.open")
    @patch("src.img_to_text_converter.pytesseract.image_to_string", return_value="Image content")
    def test_picture_to_text_valid(self, mock_image_to_string, mock_open):
        result = ImgToTextConverter._ImgToTextConverter__picture_to_text(Path("test.jpg"))
        self.assertEqual(result, "Image content")

    @patch("src.img_to_text_converter.Image.open", side_effect=UnidentifiedImageError)
    def test_picture_to_text_invalid_image(self, mock_open):
        with self.assertRaises(ValueError):
            ImgToTextConverter._ImgToTextConverter__picture_to_text(Path("test.jpg"))

    @patch("src.img_to_text_converter.Image.open", side_effect=Exception("Unexpected error"))
    def test_picture_to_text_runtime_error(self, mock_open):
        with self.assertRaises(RuntimeError):
            ImgToTextConverter._ImgToTextConverter__picture_to_text(Path("test.jpg"))


if __name__ == "__main__":
    unittest.main()















