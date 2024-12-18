import unittest
import json

from src.TextAssembler import TextAssembler
from src.TextClassifier import ClassifiedText


class TestTextAssembler(unittest.TestCase):
    def test_integration(self):
        with open('../statics/model_training_data/roadto/10-ch1.json', 'r') as file:
            training_data = json.load(file)
        
        classified_text = [ClassifiedText(td['label'], td['text']) for td in training_data]
        text_assembler = TextAssembler()
        text_assembler.process_classified_text(classified_text)
        text_assembler.__save_chapter()
        print(text_assembler.chapter.title)
        print(text_assembler.chapter.text)
        
        