from sklearn.cluster import AgglomerativeClustering
import numpy as np
from typing import List, Dict

from TextExtractor import FeaturedWord


class FeaturedBlock:
    def __init__(self, featured_words: List[FeaturedWord]):
        self.lines_count = featured_words[len(featured_words) - 1].line_number if len(featured_words) else None
        self.words_count = len(featured_words) if len(featured_words) else None
        self.left_position = min(featured_word.left_position for featured_word in featured_words) if len(featured_words) else None
        self.top_position = min(featured_word.top_position for featured_word in featured_words) if len(featured_words) else None
        self.font_size = round(sum([featured_word.box_height for featured_word in featured_words]) / self.words_count)
        self.level = round(sum([featured_word.level for featured_word in featured_words]) / self.words_count)
        self.width = (
            max(featured_word.left_position + featured_word.box_width for featured_word in featured_words) - self.left_position
            if featured_words else None
        )
        self.height = (
            max(word.top_position + word.box_height for word in featured_words) - self.top_position
            if featured_words else None
        )

        self.paragraphs = self.__extract_paragraphs(featured_words)
        self.paragraphs_count = len(self.paragraphs)

    @staticmethod
    def __extract_paragraphs(featured_words: List[FeaturedWord]) -> List[str]:
        paragraphs = {}

        for featured_word in featured_words:
            if not featured_word.paragraph_number in paragraphs:
                paragraphs[featured_word.paragraph_number] = featured_word.text_content
            else:
                paragraphs[featured_word.paragraph_number] += ' ' + featured_word.text_content

        return paragraphs.values()
    
    def __str__(self) -> str:
        paragraphs_text = '\n'.join(self.paragraphs)
        return (
            f'Level: {self.level}\n'
            f'Font Size: {self.font_size}\n'
            f'Bounding Box - Left: {self.left_position}, Top: {self.top_position}, Width: {self.width}, Height: {self.height}\n'
            f'Lines Count: {self.lines_count}\n'
            f'Words Count: {self.words_count}\n'
            f'Paragraphs:\n{paragraphs_text}'
        )


class TextClassifier:
    def classify_page(self, page_featured_words: List[FeaturedWord]) -> List[FeaturedBlock]:
        filtered_page_featured_words = self.__filter_page_featured_words(page_featured_words=page_featured_words)
        grouped_featured_words_by_the_blocks_page =  self.__group_featured_words_by_the_block(filtered_page_featured_words)
        featured_blocks_page = [FeaturedBlock(grouped_featured_words_by_the_block) for grouped_featured_words_by_the_block in grouped_featured_words_by_the_blocks_page]
        return featured_blocks_page
    
    @staticmethod
    def __filter_page_featured_words(page_featured_words: List[FeaturedWord]) -> List[FeaturedWord]:
        return filter(lambda featured_word: featured_word.text_content.strip(), page_featured_words)
    
    @staticmethod
    def __group_featured_words_by_the_block(featured_featured_text_page: List[FeaturedWord]) -> List[List[FeaturedWord]]:
        current_block_of_featured_text: List[FeaturedWord] = []
        grouped_featured_text_by_the_block: List[List[FeaturedWord]] = []
        for featured_text in featured_featured_text_page:
            if len(current_block_of_featured_text) == 0:
                current_block_of_featured_text.append(featured_text)
            elif featured_text.block_number == current_block_of_featured_text[0].block_number:
                current_block_of_featured_text.append(featured_text)
            else:
                grouped_featured_text_by_the_block.append(current_block_of_featured_text)
                current_block_of_featured_text = [featured_text]

        grouped_featured_text_by_the_block.append(current_block_of_featured_text)
        return grouped_featured_text_by_the_block
