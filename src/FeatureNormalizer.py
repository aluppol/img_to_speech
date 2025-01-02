from typing import Any
from copy import deepcopy
import pickle
from pathlib import Path
from functools import singledispatchmethod

from TextPreprocessor import FeaturedBlock, FeaturedPage, FeaturedBook


class BookFeatureNormalizerManager():
    class BookFeatureNormalizer():
        def __init__(self):
            self.min_font_size = float('inf')
            self.max_font_size = float('-inf')

            self.page_left = float('inf')
            self.page_right = float('-inf')
            self.page_top = float('inf')
            self.page_bottom = float('-inf')\

            self.min_level = float('inf')
            self.max_level = float('-inf')

            self.min_words_count = float('inf')
            self.max_words_count = float('-inf')

            self.min_lines_count = float('inf')
            self.max_lines_count = float('-inf')

            self.min_paragraphs_count = float('inf')
            self.max_paragraphs_count = float('-inf')

        @property
        def page_width(self):
            return self.page_right - self.page_left
        
        @property
        def page_height(self):
            return self.page_bottom - self.page_top
        
        @singledispatchmethod
        def train(self, featured_block: Any):
            raise TypeError(f'Unsupported type "{type(featured_block)}" for normalization')

        @train.register
        def _(self, featured_block: FeaturedBlock) -> None:
            self.min_font_size = min(self.min_font_size, featured_block.font_size)
            self.max_font_size = max(self.max_font_size, featured_block.font_size)

            self.page_left = min(self.page_left, featured_block.left_position)
            self.page_right = max(self.page_right, featured_block.left_position + featured_block.width)
            self.page_top = min(self.page_top, featured_block.top_position)
            self.page_bottom = max(self.page_bottom, featured_block.top_position + featured_block.width)

            self.min_level = min(self.min_level, featured_block.level)
            self.max_level = max(self.max_level, featured_block.level)

            self.min_words_count = min(self.min_words_count, featured_block.words_count)
            self.max_words_count = max(self.max_words_count, featured_block.words_count)
            
            self.min_lines_count = min(self.min_lines_count, featured_block.lines_count)
            self.max_lines_count = max(self.max_lines_count, featured_block.lines_count)

            self.min_paragraphs_count = min(self.min_paragraphs_count, featured_block.paragraphs_count)
            self.max_paragraphs_count = max(self.max_paragraphs_count, featured_block.paragraphs_count)

        @train.register
        def _(self, featured_page: FeaturedPage) -> None:
            for featured_block in featured_page:
                self.train(featured_block)

        @train.register
        def _(self, featured_book: FeaturedBook) -> None:
            for featured_page in featured_book:
                self.train(featured_page)

        @singledispatchmethod
        def normalize(self, featured_block: Any):
            raise TypeError(f'Unsupported type "{featured_block}" for normalization')
        
        @normalize.register
        def _(self, featured_block: FeaturedBlock) -> FeaturedBlock:
            normalized_block = deepcopy(featured_block)

            normalized_block.left_position = self.__normalize_X_position(featured_block.left_position)
            normalized_block.right_position = self.__normalize_X_position(featured_block.right_position)
            normalized_block.top_position = self.__normalize_Y_position(featured_block.top_position)
            normalized_block.bottom_position = self.__normalize_Y_position(featured_block.bottom_position)

            normalized_block.width = self.__normalize_width(featured_block.width)
            normalized_block.height = self.__normalize_height(featured_block.height)

            normalized_block.font_size = self.__normalize_numeric_property(featured_block.font_size, 'font_size')
            normalized_block.level = self.__normalize_numeric_property(featured_block.level, 'level')
            normalized_block.words_count = self.__normalize_numeric_property(featured_block.words_count, 'words_count')
            normalized_block.lines_count = self.__normalize_numeric_property(featured_block.lines_count, 'lines_count')
            normalized_block.paragraphs_count = self.__normalize_numeric_property(featured_block.paragraphs_count, 'paragraphs_count')

            return normalized_block
        
        @normalize.register
        def _(self, featured_page: FeaturedPage) -> FeaturedPage:
            normalized_featured_page = FeaturedPage()
            for featured_block in featured_page:
                normalized_featured_page.append(self.normalize(featured_block))
            return normalized_featured_page

        @normalize.register
        def _(self, featured_book: FeaturedBook) -> FeaturedBook:
            normalized_featured_book = FeaturedBook()
            for featured_page in featured_book:
                normalized_featured_book.append(self.normalize(featured_page))
            return normalized_featured_book
        
        def __normalize_X_position(self, position: int):
            if self.page_left == self.page_right or self.page_width == 0:
                return 0
            
            return (position - self.page_left) / (self.page_width)
        
        def __normalize_Y_position(self, position: int):
            if self.page_top == self.page_bottom or self.page_height == 0:
                return 0
            
            return (position - self.page_top) / (self.page_height)
            
        def __normalize_width(self, width: int):
            if self.page_width == 0:
                return 0
            
            return width / self.page_width
        
        def __normalize_height(self, height: int):
            if self.page_height == 0:
                return 0
            
            return height / self.page_height
        
        def __normalize_numeric_property(self, property_value: int, property_name: str):
            min_value = getattr(self, f'min_{property_name}')
            max_value = getattr(self, f'max_{property_name}')

            if min_value == max_value:
                return 0
            
            return (property_value - min_value) / (max_value - min_value)
        

    def __init__(self, book_title: str, featured_book: FeaturedBook, dir_path_to_normalizers='statics/feature_normalizers'):
        self.book_title = book_title
        self.dir_path_to_normalizers = dir_path_to_normalizers

        if self.__is_normalizer_for_book_exists():
            self.normalizer: BookFeatureNormalizerManager.BookFeatureNormalizer = self.load_normalizer_for_book()
        else:
            self.initiate_normalizer(featured_book)
    
    def initiate_normalizer(self, featured_book: FeaturedBook):
        self.normalizer = self.BookFeatureNormalizer()
        self.train_normalizer_with_preprocessed_book(featured_book)
        self.save_feature_normalizer_for_book()

    def train_normalizer_with_preprocessed_book(self, featured_book: FeaturedBook) -> None:
        self.normalizer.train(featured_book)

    def save_feature_normalizer_for_book(self) -> None:
        with open(f'{self.dir_path_to_normalizers}/{self.book_title}.pkl', 'wb') as file:
            pickle.dump(self.normalizer, file)

    def load_normalizer_for_book(self) -> None:
        with open(f"{self.dir_path_to_normalizers}/{self.book_title}.pkl", 'rb') as file:
            return pickle.load(file)
        
    def __is_normalizer_for_book_exists(self) -> bool:
        path = Path(f'{self.dir_path_to_normalizers}/{self.book_title}.pkl')
        return path.exists() and path.is_file()
    