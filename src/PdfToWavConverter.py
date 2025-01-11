from typing import List
import pickle
import traceback
from pathlib import Path

from TextExtractor import PdfTextExtractor
from TextPreprocessor import TextPreprocessor, FeaturedBook
from TextClassifier import LabeledFeaturedBlock, TextClassifier
from LabelTransformer import LabelTransformer
from TextAssembler import TextAssembler
from TextVocalizer import TextToSpeech, TextToSpeechPipeline
from FeatureNormalizer import BooksFeatureNormalizersManager
from utils import get_file_name_from_path


class PdfToWavConverter:
    def __init__(self, text_classifier_model_path='src/models/img_to_speech-book_text_classifier_model'):
        self.__text_classifier = TextClassifier(text_classifier_model_path)

    def generate_training_data(self, pdf_path: str, path_with_no_extension: str):
        label_transformer = LabelTransformer()
        text_extractor = PdfTextExtractor()
        text_preprocessor = TextPreprocessor()
        books_feature_normalizers_manager = BooksFeatureNormalizersManager()

        featured_words_page = next(text_extractor.extract(pdf_path))
        featured_page = text_preprocessor.preprocess_page(featured_words_page)
        feature_normalizer = books_feature_normalizers_manager.load_normalizer_for_book('roadto')
        normalized_featured_page = feature_normalizer.normalize(featured_page)
            
        labeled_normalized_featured_page = self.__text_classifier.classify_featured_page(normalized_featured_page)
    
        for labeled_normalized_featured_block in labeled_normalized_featured_page:
            print(labeled_normalized_featured_block)
            labeled_normalized_featured_block.label = str(label_transformer.to_str(labeled_normalized_featured_block.label)).split('.')[1]

        self.save_training_data(labeled_normalized_featured_page, path_with_no_extension)

    def train_text_classifier(self, training_data_dir: str, model_dir: str, epochs=5, loss_limit=0.5):
        text_classifier = TextClassifier(model_dir)
            
        text_classifier.train_model(training_data_dir, epochs=epochs, loss_limit=loss_limit)
        if not text_classifier.model_dir.exists():
            text_classifier.model_dir.mkdir()
        text_classifier.save_model(text_classifier.model_dir)

    def save_training_data(self, data: List[LabeledFeaturedBlock], file_path: str):
        self.save_to_pkl(data, file_path)
    
    @staticmethod
    def save_to_pkl(data, path_with_no_ext: str):
        with open(f'{path_with_no_ext}.pkl', 'wb') as f:
            pickle.dump(data, f)

    def pdf_to_voice_pipeline(self, pdf_file_path: str, mp3_folder_path: str):
        text_extractor = PdfTextExtractor()
        text_assembler = TextAssembler()

        model_path = Path('src/models/img_to_speech-book_text_classifier_model')
        if not model_path.exists() or not any(model_path.iterdir()):
            model_path = 'aluppol/img_to_speech-book_text_classifier'
        text_classifier = TextPreprocessor(model_path)

        try:
            for featured_text_page in text_extractor.extract(pdf_file_path, 40, 42):
                classified_text_page = text_classifier.classify_featured_text(featured_text_page)
                chapters = []
                for chapter in text_assembler.process_classified_text(classified_text_page):
                    print(f'{chapter.title} ... done')
                    chapters.append(chapter)
                text_assembler.save_chapter()
                chapter = text_assembler.chapter
                print(f'{chapter.title} ... done')
                print(chapter.title)
                print(chapter.text)
                
        except Exception as e:
            print(f'Error: {e}')
            traceback.print_exc()

    def convert_pdf_to_wav(self, pdf_path: str, wav_output_dir_path: str):
        text_extractor = PdfTextExtractor()
        text_preprocessor = TextPreprocessor()
        books_feature_normalizers_manager = BooksFeatureNormalizersManager()

        featured_words_pages = [featured_words_page for featured_words_page in text_extractor.extract(pdf_path)]
        featured_book = FeaturedBook([text_preprocessor.preprocess_page(featured_words_page) for featured_words_page in featured_words_pages])
        feature_normalizer = books_feature_normalizers_manager.load_or_init_normalizer('roadto', featured_book)   # get_file_name_from_path(pdf_path)
        normalized_featured_book = feature_normalizer.normalize(featured_book)
        for page in normalized_featured_book:
            for block in page:
                print(block)
        
