from typing import Any
from copy import deepcopy
import pickle
from pathlib import Path
from functools import singledispatchmethod

from TextClassifier import FeaturedBlock, FeaturedPage, FeaturedBook


class BookFeatureNormalizerManager():
    class BookFeatureNormalizer():
        def __init__(self):
            self.min_font_size = float('inf')
            self.max_font_size = float('-inf')
            self.page_left = float('inf')
            self.page_right = float('-inf')
            self.page_top = float('inf')
            self.page_bottom = float('-inf')

        @property
        def width(self):
            return self.page_right - self.page_left
        
        @property
        def height(self):
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
            normalized_block.font_size = (self.max_font_size - featured_block.font_size) / (self.max_font_size - self.min_font_size)
            normalized_block.left_position = (featured_block.left_position - self.page_left) / (self.width)
            normalized_block.top_position = (featured_block.top_position - self.page_top) / (self.height)
            normalized_block.width = featured_block.width / self.width
            normalized_block.height = featured_block.height / self.height
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
    