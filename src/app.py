from typing import List
import pickle
import traceback
from pathlib import Path

from TextExtractor import PdfTextExtractor
from TextPreprocessor import TextPreprocessor, FeaturedBook
from TextClassifier import LabeledFeaturedBlock, TextClassifier
from LabelTransformer import LabelTransformer
from TextAssembler import TextAssembler
from TextVocalizer import TextToSpeech, TextToSpeechPipeline
from FeatureNormalizer import BookFeatureNormalizerManager
from utils import get_file_name_from_path


def generate_training_data(pdf_path: str, path_with_no_extention: str):
  label_transformer = LabelTransformer()
  text_extractor = PdfTextExtractor()
  text_preprocessor = TextPreprocessor()
  featured_words_pages = [featured_words_page for featured_words_page in text_extractor.extract(pdf_path)]
  [featured_page] = [text_preprocessor.preprocess_page(feautured_words_page) for feautured_words_page in featured_words_pages]
  feature_normalizer = BookFeatureNormalizerManager('roadto', featured_page).normalizer   # get_file_name_from_path(pdf_path)
  normalized_featured_page = feature_normalizer.normalize(featured_page)
  
  text_classifier = TextClassifier('src/models/img_to_speech-book_text_classifier_model')
    
  labeled_normalized_fetured_page = text_classifier.classify_featured_page(normalized_featured_page)
  for labeled_normalized_fetured_block in labeled_normalized_fetured_page:
    print(labeled_normalized_fetured_block)
    labeled_normalized_fetured_block.label = str(label_transformer.to_str(labeled_normalized_fetured_block.label)).split('.')[1]

  save_training_data(labeled_normalized_fetured_page, path_with_no_extention)

def train_text_classifier(training_data_dir: str, model_dir: str, epochs=5, loss_limit=0.5):
  text_classifier = TextClassifier(model_dir)
    
  text_classifier.train_model(training_data_dir, epochs=epochs, loss_limit=loss_limit)
  if not text_classifier.model_dir.exists():
    text_classifier.model_dir.mkdir()
  text_classifier.save_model(text_classifier.model_dir)

def save_training_data(data: List[LabeledFeaturedBlock], file_path: str):
  save_to_pkl(data, file_path)
  
def save_to_pkl(data, path_with_no_ext: str):
  with open(f'{path_with_no_ext}.pkl', 'wb') as f:
    pickle.dump(data, f)

def pdf_to_voice_pipeline(pdf_file_path: str, mp3_folder_path: str):
  text_extractor = PdfTextExtractor()
  text_assembler = TextAssembler()

  model_path = Path('src/models/img_to_speech-book_text_classifier_model')
  if not model_path.exists() or not any(model_path.iterdir()):
    model_path = 'aluppol/img_to_speech-book_text_classifier'
  text_classifier = TextPreprocessor(model_path)

  try:
    for featured_text_page in text_extractor.extract(pdf_file_path, 40, 42):
      classified_text_page = text_classifier.classify_featured_text(featured_text_page)
      chapters = []
      for chapter in text_assembler.process_classified_text(classified_text_page):
          print(f'{chapter.title} ... done')
          chapters.append(chapter)
      text_assembler.save_chapter()
      chapter = text_assembler.chapter
      print(f'{chapter.title} ... done')
      print(chapter.title)
      print(chapter.text)
          
  except Exception as e:
      print(f'Error: {e}')
      traceback.print_exc()


def convert_pdf_to_wav(pdf_path: str, wav_output_dir_path: str):
  text_extractor = PdfTextExtractor()
  text_preprocessor = TextPreprocessor()
  featured_words_pages = [featured_words_page for featured_words_page in text_extractor.extract(pdf_path)]
  featured_book = FeaturedBook([text_preprocessor.preprocess_page(feautured_words_page) for feautured_words_page in featured_words_pages])
  feature_normalizer = BookFeatureNormalizerManager('roadto', featured_book).normalizer   # get_file_name_from_path(pdf_path)
  normalized_featured_book = feature_normalizer.normalize(featured_book)
  for page in normalized_featured_book:
    for block in page:
      print(block)
     

if __name__ == '__main__':
  # pdf_to_voice_pipeline('statics/roadto.pdf', 'statics/output_audio')
  generate_training_data('statics/books/roadto_9.pdf', 'statics/model_training_data/roadto/01-page-9-introduction')
  # train_text_classifier('statics/model_training_data/roadto', 'src/models/img_to_speech-book_text_classifier', loss_limit=4)
  # train_text_to_speech('src/models/img_to_speech-text_to_speech_model')
  # convert_pdf_to_wav('statics/books/roadto_9.pdf', 'statics/output_audio')
