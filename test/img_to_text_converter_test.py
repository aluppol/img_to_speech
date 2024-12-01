import unittest
from unittest.mock import patch
import os

from img_to_text_converter import ImgToTextConverter


class TestImgToTextConverter(unittest.TestCase):

    @patch('builtins.open', create=True)
    def test_pdf_conversion(self):
        """
        Test converting a PDF file to text.
        """
        # Assuming there's a sample PDF file to test with
        pdf_file_path = 'statics/Albert - Cover Letter.docx.pdf'
        expected_text = "This is some text in the PDF file."

        # Here we would mock the PdfReader functionality or use a real sample PDF for testing
        # For simplicity, we're simulating expected behavior
        # Ensure the file is valid and can be processed as a PDF
        result = ImgToTextConverter.convert(pdf_file_path)
        self.assertIn("This is some text", result)  # Basic check for expected content

    @patch('PIL.Image.open')
    def test_image_conversion(self):
        """
        Test converting a JPG image to text using Tesseract OCR.
        """
        # Assuming there's a sample JPG file to test with
        image_file_path = 'statics/photo_5221994802856441008_y.jpg'
        expected_text = "This is extracted from an image."

        # Mocking the image text conversion
        with patch('pytesseract.image_to_string', return_value=expected_text):
            result = ImgToTextConverter.convert(image_file_path)
            print(result)
            self.assertEqual(result, expected_text)  # Check that the returned text matches expected


if __name__ == '__main__':
    unittest.main()
