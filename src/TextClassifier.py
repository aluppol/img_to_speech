from transformers import BertModel, BertTokenizer, PreTrainedModel, PretrainedConfig
import heapq
import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import json

from src.Text_Extractor_Outdated import FeaturedText
from LabelTransformer import LabelTransformer
from LabelTransformer import Label


class TrainingData(FeaturedText):
  label: str


class ClassifiedText:
  label: int
  text: str

  def __init__(self, label: int, text: str):
    self.label = label
    self.text = text


class TextClassifierModelConfig(PretrainedConfig):
    def __init__(
        self,
        bert_model_name: str = None,
        num_numeric_features: int = None,
        num_classes: int = None,
        **kwargs,
      ):
        super().__init__(**kwargs)
        if not bert_model_name:
          bert_model_name = "bert-base-uncased"
        if not num_numeric_features:
          num_numeric_features = 7
        if not num_classes:
          num_classes = len(Label) + 1

        self.bert_model_name = bert_model_name
        self.num_numeric_features = num_numeric_features
        self.num_classes = num_classes

class TextClassifierModel(PreTrainedModel):
  config_class = TextClassifierModelConfig

  def __init__(
      self,
      config=TextClassifierModelConfig(),
    ):
    super(TextClassifierModel, self).__init__(config)

    # Pre-trained BERT for text embeddings
    self.bert = BertModel.from_pretrained(config.bert_model_name)
    self.numeric_features = nn.Linear(config.num_numeric_features, 128)
    self.combined_layer = nn.Linear(self.bert.config.hidden_size + 128, 256)
    self.output_layer = nn.Linear(256, config.num_classes)
    self.relu = nn.ReLU()

  def forward(self, text: List[str], numeric_features: List[List[float]]):
    # Text embeddings from BERT
    bert_output = self.bert(**text).pooler_output

    # Numeric feature transformation
    numeric_transformed = self.relu(self.numeric_features(numeric_features))

    # Combine both features
    combined = torch.cat((bert_output, numeric_transformed), dim=1)   # Shape: (batch_size, 896)
    combined = self.relu(self.combined_layer(combined))

    # Output prediction
    return self.output_layer(combined)

class TextClassifier:
  def __init__(
      self,
      model_dir: str,
      bert_model_name: str = None,
      num_numeric_features: int = None,
      num_classes: int = None,
    ):
    self.model_dir = Path(model_dir)
    self.scaler = MinMaxScaler()
    self.loss_fn = nn.CrossEntropyLoss()

    # Check if the model exists; otherwise, initialize a new one
    try:
      print(f"Loading model from {model_dir} ... loading")
      self.__load_model(model_dir)
    except Exception as e:
      print(f'Failed to load: {e}')
      print(f"Initializing a new model...")
      self.__create_model(bert_model_name, num_numeric_features, num_classes)

    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

  def predict(self, text_data: List[str], numeric_features: List[List[float]]):
    if len(text_data) != len(numeric_features):
      raise ValueError("Mismatch: text_data and numeric_features must have the same length.")

    self.model.eval()
    with torch.no_grad():
      encoded_text = self.tokenizer(text_data, return_tensors='pt', padding=True, truncation=True)
      numeric_features_tensor = torch.tensor(numeric_features, dtype=torch.float32)

      # Ensure tensors are on the same device as the model
      device = next(self.model.parameters()).device
      encoded_text = {key: val.to(device) for key, val in encoded_text.items()}
      numeric_features_tensor = numeric_features_tensor.to(device)

      prediction = self.model(encoded_text, numeric_features_tensor)
      return torch.argmax(prediction, dim=1).tolist()
    
  def save_model(self, save_dir: Path):
    if not save_dir.exists():
      save_dir.mkdir(parents=True, exist_ok=True)
    self.model.save_pretrained(save_dir)
    self.tokenizer.save_pretrained(save_dir)

  def preprocess_input(self, featured_text: List[FeaturedText]) -> Tuple[List[str], np.ndarray]:
    text_data = [fte['text'] for fte in featured_text]

    numeric_features = [
      [fte['size'], fte['flags'], fte['page']] + list(fte['bbox'])
      for fte in featured_text
    ]

    normalized_features = self.scaler.fit_transform(numeric_features)

    return text_data, normalized_features
  
  def train_model(self, training_dataset_path: str, epochs=5, loss_limit=0.5):
      training_file_paths = ([training_dataset_path]
        if Path(training_dataset_path).is_file()
        else [str(file) for file in Path(training_dataset_path).iterdir() if file.is_file() and file.suffix == '.json'])
      label_transformer = LabelTransformer()
      training_queue = [(-100, path) for path in sorted(training_file_paths)]
      while len(training_queue) > 0:
        last_loss, dataset_path = heapq.heappop(training_queue)
        last_loss = -last_loss  # invert the sign from min heap to return to normal form
        print(f'Dataset f{dataset_path} ... processing')
        with open(dataset_path, 'r') as json_file:
          training_dataset: List[TrainingData] = json.load(json_file)
        labels_str = [row["label"] for row in training_dataset]
        labels = [label_transformer.to_int(label) for label in labels_str]
        text_set, feature_set = self.preprocess_input(training_dataset)
        loss = self.__train_model(text_set, feature_set, labels, epochs=epochs)
        print(f'Dataset f{dataset_path} ... done ... from {last_loss} to {loss}')
        if loss > loss_limit:
          heapq.heappush(training_queue, (-loss, dataset_path)) # invert sign to make min heap

      print('Training ... completed')
  
  def classify_featured_text(self, featured_text: List[FeaturedText]) -> List[ClassifiedText]:
    text, numeric_features = self.preprocess_input(featured_text)
    labels = self.predict(text, numeric_features)
    
    result: List[ClassifiedText] = []

    for i in range(len(text)):
      result.append(ClassifiedText(label=labels[i], text=text[i]))

    return result


  def __train_model(self, text_data: List[str], numeric_features: np.ndarray, labels: List[int], epochs=5):
    self.model.train()
    for epoch in range(epochs):
      # Tokenize text data
      encoded_text = self.tokenizer(text_data, return_tensors='pt', padding=True, truncation=True)
      numeric_features_tensor = torch.tensor(numeric_features, dtype=torch.float32)
      labels_tensor = torch.tensor(labels, dtype=torch.int64)

      # Forward pass
      outputs = self.model(encoded_text, numeric_features_tensor)
      loss = self.loss_fn(outputs, labels_tensor)

      # Backward pass and optimization
      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()

      print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")
    return loss.item()

  def __load_model(
      self,
      model_dir: str,
    ):
    self.tokenizer = BertTokenizer.from_pretrained(model_dir)
    self.model = TextClassifierModel.from_pretrained(model_dir)

  def __create_model(
      self,
      bert_model_name: str = None,
      num_numeric_features: int = None,
      num_classes: int = None,
    ):
    config = TextClassifierModelConfig(
      bert_model_name=bert_model_name,
      num_numeric_features=num_numeric_features,
      num_classes=num_classes,
    )
    self.tokenizer = BertTokenizer.from_pretrained(config.bert_model_name)
    self.model = TextClassifierModel(
      config=config
    )
