from img_to_text_converter import ImgToTextConverter
from book_to_text_converter import BookToText


if __name__ == "__main__":
    converter = ImgToTextConverter()
    book_converter = BookToText()
    try:
        # result = converter.convert("test/statics/c.jpg")
        # print("Extracted Text:\n", result)
        # result = converter.convert("test/statics/roadto_p1.pdf")
        # print("Extracted Text:\n", result)
        result = BookToText.convert("test/statics/roadto_p1-10.pdf")
        print(result)
    except Exception as e:
        print(f"Error: {e}")