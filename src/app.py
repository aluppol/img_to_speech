from typing import List
import json
import traceback
from pathlib import Path

from TextExtractor import TextExtractor
from TextClassifier import TextClassifier, TrainingData
from LabelTransformer import LabelTransformer
from TextAssembler import TextAssembler
from TextToSpeech import TextToSpeech, TextToSpeechPipeline


def generate_training_data(path: str):
  text_extractor = TextExtractor()
  label_transformer = LabelTransformer()

  model_path = 'src/models/book_text_classifier_model'
  text_classifier = TextClassifier(model_path)
    
  from_page, to_page = (102, 102)
  featured_text = text_extractor.extract('statics/roadto.pdf', from_page, to_page)
  text, num_features = text_classifier.preprocess_input(featured_text)
  labels = text_classifier.predict(text, num_features)
  for i in range(len(labels)):
    featured_text[i]['label'] = str(label_transformer.to_str(labels[i])).split('.')[1]

  safe_training_data(featured_text, path)

def train_text_classifier(training_data_dir: str, model_dir: str, epochs=5, loss_limit=0.5):
  text_classifier = TextClassifier(model_dir)
    
  text_classifier.train_model(training_data_dir, epochs=epochs, loss_limit=loss_limit)
  if not text_classifier.model_dir.exists():
    text_classifier.model_dir.mkdir()
  text_classifier.save_model(text_classifier.model_dir)

def safe_training_data(data: List[TrainingData], file_path: str):
  save_to_json(data, file_path)
  
def save_to_json(data, path: str):
  '''
  Save the given data to a JSON file at the specified path.

  :param data: The data to be saved (must be serializable to JSON).
  :param path: The file path where the JSON will be saved.
  '''
  with open(path, 'w', encoding='utf-8') as file:
    json.dump(data, file, ensure_ascii=False, indent=4)

def pdf_to_voice_pipeline(pdf_file_path: str, mp3_folder_path: str):
  text_extractor = TextExtractor()
  text_assembler = TextAssembler()

  model_path = Path('src/models/img_to_speech-book_text_classifier_model')
  if not model_path.exists() or not any(model_path.iterdir()):
    model_path = 'aluppol/img_to_speech-book_text_classifier'
  text_classifier = TextClassifier(model_path)

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

if __name__ == '__main__':
  pdf_to_voice_pipeline('statics/roadto.pdf', 'statics/output_audio')
  # generate_training_data('statics/model_training_data/roadto/change_name.json')
  # train_text_classifier('statics/model_training_data/roadto', 'src/models/img_to_speech-book_text_classifier', loss_limit=4)
  # train_text_to_speech('src/models/img_to_speech-text_to_speech_model')