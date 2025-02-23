from typing import List, Generator, Optional, Any
from functools import singledispatchmethod
import re

from TextClassifier import LabeledFeaturedBlock, LabeledFeaturedPage, LabeledFeaturedBook
from LabelTransformer import Label, label_transformer
from TextAnalyzer import TextAnalyzer


class Annotation:
    def __init__(
        self,
        text: str = '',
        reference: str = None,
    ):
        self.text = text
        self.reference = reference

    def __str__(self):
        return self.text


class AnnotatedParagraph:
    def __init__(self, text: str, annotations: List[Annotation]):
        self.text = text
        self.annotations = annotations

    def __str__(self):
        return self.text + '\n' + '\n'.join([str(annotation) for annotation in self.annotations])


class Chapter:
    def __init__(
            self,
            title: str,
            paragraphs: List[AnnotatedParagraph],
            epigraph: Optional[AnnotatedParagraph]
        ):
        self.title = title
        self.epigraph = epigraph
        self.paragraphs = paragraphs

    def __str__(self):
        return self.title + '\n' + str(self.epigraph) + '\n' + '\n'.join([str(paragraph) for paragraph in self.paragraphs])



class Book:
    def __init__(self, title: str, chapters: List[Chapter]):
        self.title = title
        self.chapters = chapters

    def __str__(self):
        return self.title + '\n' + '\n'.join([str(chapter) for chapter in self.chapters])


class TextAssembler:
    def __init__(self):
        self.__text_analyzer = TextAnalyzer()
        self.__label_transformer = label_transformer

    def assemble_the_book(self, classified_book: LabeledFeaturedBook) -> Book:
        preprocessed_classified_book = self.__preprocess_classified_book(classified_book)
        book_title = next(iter(self.__extract_text_by_label(preprocessed_classified_book, Label.TITLE)), 'Untitled book')
        chapters = self.__extract_book_chapters(preprocessed_classified_book)
        return Book(book_title, chapters)
    
    def __extract_book_chapters(self, preprocessed_classified_book: LabeledFeaturedBook) -> List[Chapter]:
        classified_chapters = self.__slice_book_to_chapters(preprocessed_classified_book)
        chapters = [self.__assemble_chapter_from_classified_chapter(classified_chapter) for classified_chapter in classified_chapters]
        return chapters
    
    def __slice_book_to_chapters(self, classified_book: LabeledFeaturedBook) -> List[LabeledFeaturedBook]:
        classified_chapters: List[LabeledFeaturedBook] = []
        chapter_start_index = None

        for i in range(len(classified_book)):
            chapter_title_from_page = self.__extract_chapter_title_from_page(classified_book[i])
            if chapter_title_from_page:
                if chapter_start_index is not None:
                    classified_chapter = LabeledFeaturedBook(classified_book[chapter_start_index:i])
                    classified_chapters.append(classified_chapter)
                chapter_start_index = i

        if chapter_start_index is not None:
            last_classified_chapter = LabeledFeaturedBook(classified_book[chapter_start_index:])
            classified_chapters.append(last_classified_chapter)

        return classified_chapters
    
    def __assemble_chapter_from_classified_chapter(self, classified_chapter: LabeledFeaturedBook) -> Chapter:
        title = self.__extract_text_by_label(classified_chapter, Label.CHAPTER_TITLE)[0]
        epigraph = self.__join_annotated_paragraphs(self.__extract_annotated_paragraphs(classified_chapter, Label.EPIGRAPH))
        paragraphs = self.__extract_annotated_paragraphs(classified_chapter, Label.CHAPTER_TEXT)
        return Chapter(title, paragraphs, epigraph)

    def __preprocess_classified_book(self, classified_book: LabeledFeaturedBook) -> LabeledFeaturedBook:
        preprocessed_pages: List[LabeledFeaturedPage] = []

        for page in classified_book:
            preprocessed_pages.append(self.__preprocess_classified_page(page))
        return LabeledFeaturedBook(preprocessed_pages)
    
    @singledispatchmethod
    def __extract_annotated_paragraphs(self, classified_text: Any, label: Label) -> List[AnnotatedParagraph]:
        raise TypeError(f'Unsupported type "{type(classified_text)}" for text extraction') 
    
    @__extract_annotated_paragraphs.register
    def _(self, classified_book: LabeledFeaturedBook, label: Label) -> List[AnnotatedParagraph]:
        pages_paragraphs: List[List[AnnotatedParagraph]] = [self.__extract_annotated_paragraphs(page, label) for page in classified_book]
        paragraphs: List[AnnotatedParagraph] = []
        for page_paragraphs in pages_paragraphs:
            if not len(page_paragraphs):
                continue

            if len(paragraphs) == 0:
                paragraphs.extend(page_paragraphs)
            else:
                last_paragraph_of_the_page = paragraphs.pop()
                first_paragraph_of_next_page = page_paragraphs.pop(0)
                if self.__text_analyzer.is_the_same_paragraph(last_paragraph_of_the_page.text, first_paragraph_of_next_page.text):
                    paragraphs.append(self.__join_annotated_paragraphs([last_paragraph_of_the_page, first_paragraph_of_next_page]))
                else:
                    paragraphs.extend([last_paragraph_of_the_page, first_paragraph_of_next_page])
                paragraphs.extend(page_paragraphs[1:])

        return paragraphs
                
        
    @__extract_annotated_paragraphs.register
    def _(self, classified_page: LabeledFeaturedPage, label: Label) -> List[AnnotatedParagraph]:
        label_paragraphs = self.__extract_paragraphs_by_label(classified_page, label)
        page_annotations = self.__extract_paragraphs_by_label(classified_page, Label.ANNOTATION)
        annotations: List[str] = []
        annotated_paragraphs: List[AnnotatedParagraph] = []

        for annotation in page_annotations:
            annotations.extend(self.__text_analyzer.split_text_to_paragraphs(annotation))

        for paragraph in label_paragraphs:
            current_paragraph_annotations: List[str] = []
            unused_annotations: List[str] = []

            for annotation in annotations:
                if self.__text_analyzer.is_annotation_belong_to_the_text(paragraph, annotation):
                    current_paragraph_annotations.append(annotation)
                else:
                    unused_annotations.append(annotation)

            annotated_paragraphs.append(AnnotatedParagraph(paragraph, current_paragraph_annotations))
            annotations = unused_annotations

        return annotated_paragraphs
            

    @singledispatchmethod
    def __extract_paragraphs_by_label(self, classified_book: Any, label: Label) -> List[str]:
        raise TypeError(f'Unsupported type "{type(classified_book)}" for text extraction') 
    
    @__extract_paragraphs_by_label.register
    def _(self, classified_book: LabeledFeaturedBook, label: Label) -> List[str]:
        paragraphs_by_label = []
        for page in classified_book:
            paragraphs_by_label.extend(self.__extract_text_by_label(page, label))
        return paragraphs_by_label
    
    @__extract_paragraphs_by_label.register
    def _(self, classified_page: LabeledFeaturedPage, label: Label) -> List[str]:
        paragraphs_by_label = []
        for block in classified_page:
            if block.label == self.__label_transformer.to_int(label):
                paragraphs_by_label.extend(block.paragraphs)
        return paragraphs_by_label

    @singledispatchmethod
    def __extract_text_by_label(self, classified_book: Any, label: Label) -> List[str]:
        raise TypeError(f'Unsupported type "{type(classified_book)}" for text extraction') 
    
    @__extract_text_by_label.register
    def _(self, classified_book: LabeledFeaturedBook, label: Label) -> List[str]:
        text_by_label = []
        for page in classified_book:
            text_by_label.extend(self.__extract_text_by_label(page, label))
        return text_by_label
    
    @__extract_text_by_label.register
    def _(self, classified_page: LabeledFeaturedPage, label: Label) -> List[str]:
        text_by_label = []
        for block in classified_page:
            if block.label == self.__label_transformer.to_int(label):
                text_by_label.append(block.text)
        return text_by_label

    def __extract_chapter_title_from_page(self, page: LabeledFeaturedPage) -> str:
        title = next(iter(self.__extract_text_by_label(page, Label.CHAPTER_TITLE)), "")
        if re.search(r'[A-Za-z]', title):
            return title
        else:
            return ""

    
    @staticmethod
    def __preprocess_classified_page(classified_page: LabeledFeaturedPage) -> LabeledFeaturedPage:
        preprocessed_blocks: List[LabeledFeaturedBlock] = []

        for block in classified_page:
            last_block = None if not len(preprocessed_blocks) else preprocessed_blocks[len(preprocessed_blocks) - 1]

            if last_block and block.label == last_block.label:
                last_block.paragraphs.extend(block.paragraphs)
            else:
                preprocessed_blocks.append(block)

        return LabeledFeaturedPage(preprocessed_blocks, [block.label for block in preprocessed_blocks])
    
    @staticmethod
    def __join_annotated_paragraphs(paragraphs: List[AnnotatedParagraph], join_with = '\n') -> AnnotatedParagraph:
        texts: List[str] = [paragraph.text for paragraph in paragraphs]
        annotations: List[str] = []
        for paragraph in paragraphs:
            texts.append(paragraph.text)
            annotations.extend(paragraph.annotations)

        return AnnotatedParagraph(join_with.join(texts), annotations)
    