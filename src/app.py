from img_to_text_converter import ImgToTextConverter


if __name__ == "__main__":
    converter = ImgToTextConverter()
    try:
        result = converter.convert("test/statics/c.jpg")
        print("Extracted Text:\n", result)
        result = converter.convert("test/statics/cl.pdf")
        print("Extracted Text:\n", result)
    except Exception as e:
        print(f"Error: {e}")