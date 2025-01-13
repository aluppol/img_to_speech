import heapq
from typing import List
from pathlib import Path
import torch
import json

from TextClassifier import TextClassifier
from LabelTransformer import label_transformer
from TrainingDataGenerator import TrainingLabeledFeaturedPage


class TrainingDataset:
  def __init__(self, path: str, text_data: List[str], featured_data: List[List[float]], labels: List[int]):
    self.path = path
    self.text_data = text_data
    self.featured_data = featured_data
    self.labels = labels


class TextClassifierTrainer:
    def __init__(self, training_data_dir: str):
        self.__training_data_dir = training_data_dir
        self.__text_classifier = TextClassifier()

    def train_text_classifier(self, epochs=5, loss_limit=0.5):
        self.__conduct_training_round(self.__training_data_dir, epochs=epochs, loss_limit=loss_limit)
        if not self.__text_classifier.model_dir.exists():
            self.__text_classifier.model_dir.mkdir()
        self.__text_classifier.save_model()

    def __conduct_training_round(self, training_datasets_path: str, epochs=5, loss_limit=0.5):
      training_datasets_paths = self.__load_path_to_each_training_dataset(training_datasets_path)
      loaded_training_datasets = self.__load_training_datasets_from_paths(training_datasets_paths)
      training_queue = [(-100, dataset) for dataset in loaded_training_datasets]
      while len(training_queue) > 0:
        last_loss, training_dataset = heapq.heappop(training_queue)
        last_loss = -last_loss  # invert the sign from min heap to return to normal form
        print(f'Dataset f{training_dataset.path} ... processing')
        loss = self.__train_model_with_dataset(training_dataset, epochs=epochs)
        print(f'Dataset f{training_dataset.path} ... done ... from {last_loss} to {loss}')
        if loss > loss_limit:
          heapq.heappush(training_queue, (-loss, training_dataset)) # invert sign to make min heap

      print('Training ... completed')


    def __train_model_with_dataset(self, training_dataset: TrainingDataset, epochs=5):
        self.__text_classifier.model.train()
        for epoch in range(epochs):
        # Tokenize text data
            encoded_text = self.__text_classifier.tokenizer(training_dataset.text_data, return_tensors='pt', padding=True, truncation=True)
            numeric_features_tensor = torch.tensor(training_dataset.featured_data, dtype=torch.float32)
            labels_tensor = torch.tensor(training_dataset.labels, dtype=torch.int64)

            # Forward pass
            outputs = self.__text_classifier.model(encoded_text, numeric_features_tensor)
            loss = self.__text_classifier.loss_fn(outputs, labels_tensor)

            # Backward pass and optimization
            self.__text_classifier.optimizer.zero_grad()
            loss.backward()
            self.__text_classifier.optimizer.step()

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")
        return loss.item()

    def __load_training_datasets_from_paths(self, training_dataset_paths: List[str]) -> List[TrainingDataset]:
        traininig_datasets: List[TrainingDataset] = []
        for dataset_path in training_dataset_paths:
            with open(dataset_path, 'r') as file:
                training_labeled_featured_page = TrainingLabeledFeaturedPage(json.load(file))
        
        training_text = self.__text_classifier.extract_text_from_featured_page(training_labeled_featured_page)
        training_features = self.__text_classifier.extract_features_from_featured_page(training_labeled_featured_page)
        labels = self.__extract_lables_from_labled_featured_page(training_labeled_featured_page)

        traininig_datasets.append(TrainingDataset(dataset_path, training_text, training_features, labels))
        return traininig_datasets

    @staticmethod
    def __load_path_to_each_training_dataset(loading_path: str) -> List[str]:
        training_dataset_extention = '.json'
        file_paths = None
        if Path(loading_path).is_file():      
            file_paths = [loading_path]
        else:
            file_paths = [str(file) for file in Path(loading_path).iterdir() if file.is_file() and file.suffix == training_dataset_extention]
        
        return sorted(file_paths)
    
    @staticmethod
    def __extract_lables_from_labled_featured_page(labeled_featured_page: TrainingLabeledFeaturedPage) -> List[int]:
        labels: List[int] = []
        for labeled_featured_block in labeled_featured_page:
            labels.append(label_transformer.to_int(labeled_featured_block.label))
        return labels