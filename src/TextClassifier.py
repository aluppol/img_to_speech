from transformers import BertModel, BertTokenizer
import heapq
import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import json

from TextExtractor import FeaturedText
from LabelTransformer import LabelTransformer


class TrainingData(FeaturedText):
  label: str


class Chapter:
  def __init__(self, title: str, text: str, annotation: str):
    self.title = title
    self.text = text
    self.annotation = annotation


class TextClassifierModel(nn.Module):
  def __init__(self, bert_model_name: str, num_numeric_features: int, num_classes: int):
    super(TextClassifierModel, self).__init__()

    # Pre-trained BERT for text embeddings
    self.bert = BertModel.from_pretrained(bert_model_name)
    self.numeric_features_layer = nn.Linear(num_numeric_features, 128)
    self.combined_layer = nn.Linear(768 + 128, 256)
    self.output_layer = nn.Linear(256, num_classes)
    self.relu = nn.ReLU()

  def forward(self, text: List[str], numeric_features: List[List[float]]):
    # Text embeddings from BERT
    bert_output = self.bert(**text).pooler_output

    # Numeric feature transformation
    numeric_transformed = self.relu(self.numeric_features_layer(numeric_features))

    # Combine both features
    combined = torch.cat((bert_output, numeric_transformed), dim=1)   # Shape: (batch_size, 896)
    combined = self.relu(self.combined_layer(combined))

    # Output prediction
    return self.output_layer(combined)


class TextClassifierPipeline:
  def __init__(self, model_path: str, bert_model_name: str, num_numeric_features: int, num_classes: int):
    self.model_path = Path(model_path) if Path(model_path).exists() else None
    self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    self.scaler = MinMaxScaler()
    self.loss_fn = nn.CrossEntropyLoss()
    pretrain_model = False

    # Check if the model exists; otherwise, initialize a new one
    if self.model_path:
      self.model = torch.load(self.model_path)
    else:
      self.model = TextClassifierModel(bert_model_name, num_numeric_features, num_classes)
      self.model_path = Path(model_path)
      pretrain_model = True

    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

    if pretrain_model:
      self.train_model('statics/model_training_data/roadto')

  
  def predict(self, text_data: List[str], numeric_features: List[List[float]]):
    self.model.eval()
    with torch.no_grad():
      encoded_text = self.tokenizer(text_data, return_tensors='pt', padding=True, truncation=True)
      numeric_features_tensor = torch.tensor(numeric_features, dtype=torch.float32)
      prediction = self.model(encoded_text, numeric_features_tensor)
      return torch.argmax(prediction, dim=1).tolist()
    
  def save_model(self):
    torch.save(self.model, self.model_path)


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
      self.save_model()
  
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
