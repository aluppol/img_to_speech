from src.img_to_text_converter import ImgToTextConverter


class BookToText:
  '''
    Initially I will work with pdf for simplicity using img_to_text converter to get raw text before making sense of it, multiple input formats can be added later.
  '''
  def convert(url: str) -> str:
    raw_text = ImgToTextConverter.convert(url)
    
    