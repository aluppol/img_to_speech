from sklearn.cluster import AgglomerativeClustering
import numpy as np
from typing import List

from TextExtractor import FeaturedText


class TextClassifier:
    def classify_page(self, featured_text_page: List[FeaturedText]):
        filtered_featured_text_page = self.__filter_featured_text_page(featured_text_page=featured_text_page)
        grouped_featured_text_by_the_block =  self.__group_featured_text_by_the_block(filtered_featured_text_page)
        return grouped_featured_text_by_the_block

    @staticmethod
    def __filter_featured_text_page(featured_text_page: List[FeaturedText]) -> List[FeaturedText]:
        return filter(lambda featured_text: featured_text.text_content.strip(), featured_text_page)
    
    @staticmethod
    def __group_featured_text_by_the_block(featured_featured_text_page: List[FeaturedText]) -> List[List[FeaturedText]]:
        current_block_of_featured_text: List[FeaturedText] = []
        grouped_featured_text_by_the_block: List[List[FeaturedText]] = []
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
