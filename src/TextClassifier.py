from transformers import BertModel, BertTokenizer, PreTrainedModel, PretrainedConfig
import torch
import torch.nn as nn
from typing import List
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

from TextPreprocessor import FeaturedBlock, FeaturedPage, FeaturedBook
from LabelTransformer import Label


class LabeledFeaturedBlock(FeaturedBlock):
    def __init__(self, featured_block: FeaturedBlock, label: int):
        self.__dict__.update(featured_block.__dict__)
        self.label = label

    def __str__(self) -> str:
        base_str = super().__str__()
        return f'{base_str}\nLabel: {self.label}'
    
    def __repr__(self) -> str:
      return self.__str__()
    

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
          num_numeric_features = 11
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
      model_dir='src/models/img_to_speech-book_text_classifier_model',
      model_repo='aluppol/img_to_speech-book_text_classifier',
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
    
  def save_model(self):
    if not self.model_dir.exists():
      self.model_dir.mkdir(parents=True, exist_ok=True)
    self.model.save_pretrained(self.model_dir)
    self.tokenizer.save_pretrained(self.model_dir)

  def classify_featured_book(self, featured_book: FeaturedBook) -> LabeledFeaturedBook:
    labeled_featured_book = LabeledFeaturedBook()
    for featured_page in featured_book:
      labeled_featured_page = self.classify_featured_page(featured_page)
      labeled_featured_book.append(labeled_featured_page)
  
    return labeled_featured_book

  def classify_featured_page(self, featured_page: FeaturedPage) -> LabeledFeaturedPage:
    text_by_block = self.extract_text_from_featured_page(featured_page)
    numeric_features_by_block = self.extract_features_from_featured_page(featured_page)
    labels = self.predict(text_by_block, numeric_features_by_block)
    return LabeledFeaturedPage(featured_page, labels)

  def extract_features_from_featured_page(self, featured_page: FeaturedPage) -> List[List[float]]:
    features_by_block: List[List[float]] = []

    for featured_block in featured_page:
      features_from_block = self.extract_features_from_featured_block(featured_block)
      features_by_block.append(features_from_block)

    return features_by_block
  
  @staticmethod
  def extract_text_from_featured_page(featured_page: FeaturedPage) -> List[str]:
    return [featured_block.text for featured_block in featured_page]

  @staticmethod
  def extract_features_from_featured_block(featured_block: FeaturedBlock) -> List[float]:
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
  