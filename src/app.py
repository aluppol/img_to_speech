from typing import List
import json

from TextExtractor import TextExtractor
from TextClassifier import TextClassifier, TrainingData
from LabelTransformer import Label, LabelTransformer
from TextAssembler import TextAssembler


def generate_training_data(path: str):
  text_extractor = TextExtractor()
  label_transformer = LabelTransformer()

  model_path = "src/text_classifier_model.pth"
  bert_model_name = "bert-base-uncased"
  num_numeric_features = 7 # 
  num_classes = len(Label) + 1
  text_classifier = TextClassifier(model_path, bert_model_name, num_numeric_features, num_classes)
    
  from_page, to_page = (102, 102)
  featured_text = text_extractor.extract('statics/roadto.pdf', from_page, to_page)
  text, num_features = text_classifier.preprocess_input(featured_text)
  labels = text_classifier.predict(text, num_features)
  for i in range(len(labels)):
    featured_text[i]["label"] = str(label_transformer.to_str(labels[i])).split('.')[1]

  safe_training_data(featured_text, path)

def train_model(path: str, epochs=5, loss_limit=0.5):
  model_path = "src/text_classifier_model.pth"
  bert_model_name = "bert-base-uncased"
  num_numeric_features = 7 # 
  num_classes = len(Label) + 1
  text_classifier = TextClassifier(model_path, bert_model_name, num_numeric_features, num_classes)
    
  text_classifier.train_model(path, epochs=epochs, loss_limit=loss_limit)

def safe_training_data(data: List[TrainingData], file_path: str):
  save_to_json(data, file_path)
  
def save_to_json(data, path: str):
  """
  Save the given data to a JSON file at the specified path.

  :param data: The data to be saved (must be serializable to JSON).
  :param path: The file path where the JSON will be saved.
  """
  with open(path, 'w', encoding='utf-8') as file:
    json.dump(data, file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
  # run()
  # generate_training_data('statics/model_training_data/roadto/change_name.json')
  train_model('statics/model_training_data/roadto', loss_limit=0.1)


def pdf_to_voice_pipeline(pdf_file_path: str, mp3_foler_path: str):
  text_extractor = TextExtractor()
  label_transformer = LabelTransformer()
  text_assembler = TextAssembler()

  model_path = "src/text_classifier_model.pth"
  bert_model_name = "bert-base-uncased"
  num_numeric_features = 7 # of features for one data chunk to classify
  num_classes = len(Label) + 1
  text_classifier = TextClassifier(model_path, bert_model_name, num_numeric_features, num_classes)

  try:
    for featured_text_page in text_extractor.extract(pdf_file_path):
      classified_text_page = text_classifier.classify_featured_text(featured_text_page)
      for chapter in text_assembler.process_classified_text(classified_text_page):
          print(chapter.title)
          print(chapter.text)
          print(chapter.annotation)
          
  except Exception as e:
      print(f"Error: {e}")
