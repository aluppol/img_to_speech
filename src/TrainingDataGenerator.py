from typing import Any, List
from pathlib import Path
import pymupdf
from functools import singledispatchmethod
import json

from utils import count_files_in_dir
from LabelTransformer import label_transformer
from TextExtractor import PdfTextExtractor
from TextPreprocessor import TextPreprocessor, FeaturedBlock
from TextClassifier import TextClassifier, LabeledFeaturedBlock, LabeledFeaturedPage

from FeatureNormalizer import BooksFeatureNormalizersManager


class TrainingLabeledFeaturedBlock(FeaturedBlock):
    def __init__(self, *args):
       self.init(*args)

    @singledispatchmethod
    def init(self, *args: Any):
        raise TypeError(f'Unsupported type {type(args)}')

    @init.register
    def _(self, labeled_featured_block: LabeledFeaturedBlock):
        self.__dict__.update(labeled_featured_block.__dict__)
        self.label = self.__transform_label_int_to_str(labeled_featured_block.label)

    @init.register
    def _(self, data: dict):
        self.__dict__.update(data)

    def __str__(self) -> str:
        base_str = super().__str__()
        return f'{base_str}\nLabel: {self.label}'
    
    def to_dict(self):
        return {
            "lines_count": self.lines_count,
            "words_count": self.words_count,
            "left_position": self.left_position,
            "top_position": self.top_position,
            "font_size": self.font_size,
            "level": self.level,
            "width": self.width,
            "height": self.height,
            "right_position": self.right_position,
            "bottom_position": self.bottom_position,
            "paragraphs": self.paragraphs,
            "label": self.label,
        }
    
    @staticmethod
    def __transform_label_int_to_str(label: int) -> str:
        return str(label_transformer.to_str(label)).split('.')[1]


class TrainingLabeledFeaturedPage(List[TrainingLabeledFeaturedBlock]):
    def __init__(self, labeled_page: LabeledFeaturedPage = []):
        training_blocks = [TrainingLabeledFeaturedBlock(labeled_block) for labeled_block in labeled_page]
        super().__init__(training_blocks)

    def to_json_serializable(self):
        return [block.to_dict() for block in self]


class TrainingDataGenerator:
    def __init__(
        self,
        training_data_dir='statics/model_training_data',
        books_dir='statics/books',
    ):
        self.__training_data_dir=training_data_dir
        self.__books_dir=books_dir

        self.__text_classifier = TextClassifier()
        self.__text_extractor = PdfTextExtractor()
        self.__text_preprocessor = TextPreprocessor()
        self.__books_feature_normalizers_manager = BooksFeatureNormalizersManager()

    def generate_training_data(self, book_title: str, page_number: int):
        page_bytes = self.__extract_page_from_book_as_bytes(book_title, page_number)
        featured_words_page = next(self.__text_extractor.extract(page_bytes))
        featured_page = self.__text_preprocessor.preprocess_page(featured_words_page)
        feature_normalizer = self.__books_feature_normalizers_manager.load_normalizer_for_book(book_title)
        normalized_featured_page = feature_normalizer.normalize(featured_page)
            
        labeled_normalized_featured_page = self.__text_classifier.classify_featured_page(normalized_featured_page)

        training_page = TrainingLabeledFeaturedPage(labeled_normalized_featured_page)
        training_file_path = self.__get_training_file_path(book_title, page_number)
        self.__save_training_data(training_file_path, training_page)

    def __save_training_data(self, file_path: str, training_page: TrainingLabeledFeaturedPage):
        json_data = json.dumps(training_page.to_json_serializable(), indent=4)
        if file_path:
            with open(file_path, 'w') as file:
                file.write(json_data)

    def __extract_page_from_book_as_bytes(self, book_title: str, page_number: int):
        book_path = Path(self.__get_book_path(book_title))
        book_doc = pymupdf.open(book_path)
        
        if page_number < 1 or page_number > book_doc.page_count:
            raise ValueError("Invalid page number.")
    
        page_doc = pymupdf.open()
        page_doc.insert_pdf(book_doc, from_page=page_number - 1, to_page=page_number - 1)

        page_bytes = page_doc.write()

        page_doc.close()
        book_doc.close()

        return page_bytes

    def __get_book_path(self, title: str) -> str:
        return f'{self.__books_dir}/{title}.pdf'
    
    def __get_training_file_path(self, book_title: str, page_number: str) -> str:
        training_files_dir = f'{self.__training_data_dir}/{book_title}'
        files_count = count_files_in_dir(training_files_dir)
        return f'{training_files_dir}/{str(files_count + 1).zfill(2)}_{book_title}_page_{page_number}.json'

    

        

