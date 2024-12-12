from typing import List
import json

from TextExtractor import TextExtractorPipeline
from TextClassifier import TextClassifierPipeline, TrainingData
from LabelTransformer import Label, LabelTransformer

def run():
    text_extractor = TextExtractorPipeline()
    label_transformer = LabelTransformer()

    model_path = "src/text_classifier_model.pth"
    bert_model_name = "bert-base-uncased"
    num_numeric_features = 7 # 
    num_classes = len(Label) + 1
    text_classifier = TextClassifierPipeline(model_path, bert_model_name, num_numeric_features, num_classes)

    try:
      from_page, to_page = (135, 135)
      for i in range(from_page, to_page):
        featured_text = text_extractor.extract('statics/roadto.pdf', i, i)
        text, num_features = text_classifier.preprocess_input(featured_text)
        labels = text_classifier.predict(text, num_features)
        for i in range(len(labels)):
          print(text[i], label_transformer.to_str(labels[i]))
           
           
    except Exception as e:
        print(f"Error: {e}")


def generate_training_data(path: str):
  text_extractor = TextExtractorPipeline()
  label_transformer = LabelTransformer()

  model_path = "src/text_classifier_model.pth"
  bert_model_name = "bert-base-uncased"
  num_numeric_features = 7 # 
  num_classes = len(Label) + 1
  text_classifier = TextClassifierPipeline(model_path, bert_model_name, num_numeric_features, num_classes)
    
  from_page, to_page = (102, 102)
  featured_text = text_extractor.extract('statics/roadto.pdf', from_page, to_page)
  text, num_features = text_classifier.preprocess_input(featured_text)
  labels = text_classifier.predict(text, num_features)
  for i in range(len(labels)):
    featured_text[i]["label"] = str(label_transformer.to_str(labels[i])).split('.')[1]

  safe_training_data(featured_text, path)

def train_model(path: str, rounds: int=1):
  model_path = "src/text_classifier_model.pth"
  bert_model_name = "bert-base-uncased"
  num_numeric_features = 7 # 
  num_classes = len(Label) + 1
  text_classifier = TextClassifierPipeline(model_path, bert_model_name, num_numeric_features, num_classes)
    
  text_classifier.train_model(path, epochs=1, training_set_iteration_size=1000, rounds=rounds)

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
  generate_training_data('statics/model_training_data/roadto/15-ch7.json')
  # train_model('statics/model_training_data/12112024_01.json', rounds=1)

