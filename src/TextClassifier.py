from transformers import BertModel, BertTokenizer, PreTrainedModel, PretrainedConfig
import heapq
import numpy as np
import torch
import torch.nn as nn
from typing import List
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import json

from TextPreprocessor import FeaturedBlock, FeaturedPage, FeaturedBook
from LabelTransformer import LabelTransformer, Label


class LabeledFeaturedBlock(FeaturedBlock):
    def __init__(self, featured_block: FeaturedBlock, label: str):
        self.__dict__.update(featured_block.__dict__)
        self.label = label

    def __str__(self) -> str:
        base_str = super().__str__()
        return f'{base_str}\nLabel: {self.label}'
    

class LabeledFeaturedPage(List[LabeledFeaturedBlock]):
    def __init__(self, featured_page: FeaturedPage = [], labels = []):
        if len(featured_page) != len(labels):
          raise ValueError("Mismatch: featured_page and labels must have the same length.")

        labeled_featured_blocks = []
        for i in range(len(featured_page)):
           labeled_featured_blocks.append(LabeledFeaturedBlock(featured_page[i], labels[i]))
        super().__init__(labeled_featured_blocks)


class LabeledFeaturedBook(List[LabeledFeaturedPage]):
    def __init__(self, pages: List[LabeledFeaturedPage] = []):
        super().__init__(pages)


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
          training_dataset: List[LabeledFeaturedBlock] = json.load(json_file)
        labels_str = [row["label"] for row in training_dataset]
        labels = [label_transformer.to_int(label) for label in labels_str]
        text_set, feature_set = self.preprocess_input(training_dataset)
        loss = self.__train_model(text_set, feature_set, labels, epochs=epochs)
        print(f'Dataset f{dataset_path} ... done ... from {last_loss} to {loss}')
        if loss > loss_limit:
          heapq.heappush(training_queue, (-loss, dataset_path)) # invert sign to make min heap

      print('Training ... completed')
  
  def classify_featured_book(self, featured_book: FeaturedBook) -> LabeledFeaturedBook:
    labeled_featured_book = LabeledFeaturedBook()
    for featured_page in featured_book:
      labeled_featured_page = self.classify_featured_page(featured_page)
      labeled_featured_book.append(labeled_featured_page)
  
    return labeled_featured_book

  def classify_featured_page(self, featured_page: FeaturedPage) -> LabeledFeaturedPage:
    text_by_block = self.__extract_text_from_featured_page(featured_page)
    numeric_features_by_block = self.__extract_features_from_featured_page(featured_page)
    labels = self.predict(text_by_block, numeric_features_by_block)
    return LabeledFeaturedPage(featured_page, labels)
     
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

  def __extract_text_from_featured_page(self, featured_page: FeaturedPage) -> List[str]:
    text_by_block: List[str] = []

    for featured_block in featured_page:
      text_from_block = self.__extract_text_from_featured_block(featured_block)
      text_by_block.append(text_from_block)
    
    return text_by_block

  def __extract_features_from_featured_page(self, featured_page: FeaturedPage) -> List[List[float]]:
    features_by_block: List[List[float]] = []

    for featured_block in featured_page:
      features_from_block = self.__extract_features_from_featured_block(featured_block)
      features_by_block.append(features_from_block)

  @staticmethod
  def __extract_text_from_featured_block(featured_block: FeaturedBlock) -> str:
     return '\n'.join(featured_block.paragraphs)

  @staticmethod
  def __extract_features_from_featured_block(featured_block: FeaturedBlock) -> List[float]:
     return [
        featured_block.left_position,
        featured_block.top_position,
        featured_block.right_position,
        featured_block.bottom_position,
        featured_block.height,
        featured_block.width,
        featured_block.font_size,
        featured_block.level,
        featured_block.paragraphs_count,
        featured_block.lines_count,
        featured_block.words_count,
      ]
  