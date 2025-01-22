from typing import List, Generator, Optional
import re

from TextClassifier import LabeledFeaturedBlock, LabeledFeaturedPage, LabeledFeaturedBook
from LabelTransformer import Label


class Annotation:
    def __init__(
        self,
        text: str = '',
        reference: str = None,
    ):
        self.text = text
        self.reference = reference


class Paragraph:
      def __init__(self, text: str, annotations: List[Annotation]):
          self.text = text
          self.annotations = annotations


class Chapter:
  def __init__(
        self,
        title: str,
        paragraphs: List[Paragraph],
        epigraph: Optional[Paragraph]
    ):
    self.title = title
    self.epigraph = epigraph
    self.paragraphs = paragraphs


class Book:
    def __init__(self, title: str, chapters: List[Chapter]):
        self.title = title
        self.chapters = chapters


class TextAssembler:
    def assemble_the_book(self, classified_book: LabeledFeaturedBook) -> Book:
        preprocessed_classified_book = self.__preprocess_classified_book(classified_book)
        book_title = self.__extract_text_by_label(preprocessed_classified_book, Label.TITLE)[0]
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
            chapter_title_from_page = self.__extract_chapter_title(classified_page=classified_book[i])
            if chapter_title_from_page:
                if chapter_start_index:
                    classified_chapter = LabeledFeaturedBook(classified_book[chapter_start_index:i])
                    classified_chapters.append(classified_chapter)
                chapter_start_index = i

        if chapter_start_index:
            last_classified_chapter = LabeledFeaturedBook(classified_book[chapter_start_index:])
            classified_chapters.append(last_classified_chapter)

        return classified_chapters
    
    def __assemble_chapter_from_classified_chapter(self, classified_chapter: LabeledFeaturedBook) -> Chapter:
        title = self.__extract_text_by_label(classified_chapter, Label.CHAPTER_TITLE)[0]
        epigraph = self.__join_paragraphs(self.__extract_annotated_text_by_label(classified_chapter, Label.EPIGRAPH))
        paragraphs = self.__extract_annotated_text_by_label(classified_chapter, Label.CHAPTER_TEXT)
        return Chapter(title, paragraphs, epigraph)

    def __preprocess_classified_book(self, classified_book: LabeledFeaturedPage) -> LabeledFeaturedPage:
        preprocessed_pages: List[LabeledFeaturedPage] = []

        for page in classified_book:
            preprocessed_pages.append(self.__preprocess_classified_page(page))
        return LabeledFeaturedBook(preprocessed_pages)
    
    def __extract_annotated_text_by_label(self, classified_book: LabeledFeaturedBook, label: Label) -> List[Paragraph]:
        pass

    @staticmethod
    def __extract_text_by_label(classified_book: LabeledFeaturedBook, label: Label) -> List[str]:
        text_by_label = []
        for page in classified_book:
            for block in page:
                if block.label == Label:
                    text_by_label.append(block.text)
        return text_by_label

    @staticmethod
    def __preprocess_classified_page(classified_page: LabeledFeaturedPage) -> LabeledFeaturedPage:
        preprocessed_blocks: List[LabeledFeaturedBlock] = []

        for block in classified_page:
            last_block = None if not len(preprocessed_blocks) else preprocessed_blocks[len(preprocessed_blocks) - 1]

            if last_block and block.label == last_block.label:
                last_block.paragraphs.extend(block.paragraphs)
            else:
                preprocessed_blocks.append(block)

        return LabeledFeaturedPage(preprocessed_blocks)
    
    @staticmethod
    def __join_paragraphs(paragraphs: List[Paragraph]) -> Paragraph:
        texts: List[str] = [paragraph.text for paragraph in paragraphs]
        annotations: List[str] = []
        for paragraph in paragraphs:
            texts.append(paragraph.text)
            annotations.extend(paragraph.annotations)

        return Paragraph('\n'.join(texts), annotations)





    # def process_classified_text(self, classified_text: List[LabeledFeaturedBlock]) -> Generator[Chapter, None, None]:
    #     for chunk in classified_text:
    #         match chunk.label:
    #             case Label.TITLE.value:
    #                 self.__process_title(chunk.text)
                
    #             case Label.CHAPTER_TITLE.value:
    #                 self.__process_chapter_title(chunk.text)

    #             case Label.CHAPTER_TEXT.value:
    #                 self.__process_chapter_text(chunk.text)

    #             case Label.ANNOTATION.value:
    #                 self.__process_annotation(chunk.text)
                
    #             case Label.ANNOTATION_REFERENCE.value:
    #                 self.__process_annotation_references(chunk.text)

    #             case Label.EPIGRAPH.value:
    #                 self.__process_epigraph(chunk.text)

    #             case Label.AUTHOR.value:
    #                 self.__process_author(chunk.text)

    #             case _:
    #                 pass    # add to logging all passed chunks
        
    #     if self.chapter:
    #         chapter = self.chapter
    #         self.chapter = None
    #         yield chapter

    # def save_chapter(self):
    #     chapter_text = ''
    #     for i in range(len(self.epigraphs)):
    #         chapter_text += self.epigraphs[i] + '\n'
    #         chapter_text += self.authors[i] + '\n'

    #     for sentense in self.sentenses:
    #         chapter_text += ' ' + sentense.text
    #         for reference in sentense.annotation_references:
    #             annotation = next((ann for ann in self.annotations if ann.reference == reference), None)
    #             if annotation:
    #                 chapter_text += ' ' + annotation.text
    #             else:
    #                 raise Exception(f'No reference for "{reference}" annotation found!')
                        
    #     self.chapter = Chapter(self.chapter_title, chapter_text)

    #     self.chapter_title = None
    #     self.annotation_references = []
    #     self.annotations = []
    #     self.epigraphs = []
    #     self.authors = []
    #     self.sentenses = []
    #     self.last_chunk_label = None

    # def __process_title(self, text: str):
    #     if (self.title or not self.last_chunk_label) and self.last_chunk_label == Label.TITLE:
    #         self.title += text
    #         self.last_chunk_label = Label.TITLE
    #     elif self.last_chunk_label != Label.TITLE:
    #         self.classification_mistakes.append(LabeledFeaturedBlock(Label.TITLE, text))

    # def __process_chapter_title(self, text: str):
    #     if self.last_chunk_label != Label.CHAPTER_TITLE:
    #         if self.chapter_title:
    #             self.save_chapter()
    #         self.chapter_title = text
    #         self.last_chunk_label = Label.CHAPTER_TITLE

    #     else:
    #         self.chapter_title += text


    # def __process_chapter_text(self, text: str):
    #     if self.last_chunk_label == Label.ANNOTATION_REFERENCE:
    #         ref = self.annotation_references[len(self.annotation_references) - 1]
    #         current_sentense_text = self.sentenses[len(self.sentenses) - 1].text.strip()
    #         self.sentenses[(len(self.sentenses) - 1) if current_sentense_text else (len(self.sentenses) - 2)].annotation_references.append(ref)

    #     end_of_sentense_match = re.match(r"(.*?[.!?]\s*)(.*)", text, re.DOTALL)
    #     while end_of_sentense_match:
    #         self.sentenses[len(self.sentenses) - 1].text += end_of_sentense_match.group(1)
    #         self.sentenses.append(Sentense())
    #         text = end_of_sentense_match.group(2)
    #         end_of_sentense_match = re.match(r"(.*?[.!?]\s*)(.*)", text, re.DOTALL)

    #     self.sentenses[len(self.sentenses) - 1].text += text
    #     self.last_chunk_label = Label.CHAPTER_TEXT

    # def __process_annotation(self, text: str):
    #     if self.last_chunk_label == Label.ANNOTATION_REFERENCE:
    #         self.annotations.append(Annotation(text, self.annotation_references[len(self.annotation_references) - 1]))
    #         self.annotation_references.append('')
    #     else:
    #         self.annotations[len(self.annotations) - 1].text += text

    #     self.last_chunk_label = Label.ANNOTATION

    # def __process_annotation_references(self, text: str):
    #     if self.last_chunk_label == Label.ANNOTATION_REFERENCE:
    #         self.annotation_references[len(self.annotation_references) - 1] += re.sub(r"[^a-zA-Z0-9]", "", text)
    #     else:
    #         self.annotation_references.append(re.sub(r"[^a-zA-Z0-9]", "", text))
    #         self.last_chunk_label = Label.ANNOTATION_REFERENCE

    # def __process_epigraph(self, text: str):
    #     if self.last_chunk_label == Label.EPIGRAPH:
    #         self.epigraphs[len(self.epigraphs) - 1] += text
    #     else:
    #         self.epigraphs.append(text)
        
    #     self.last_chunk_label = Label.EPIGRAPH

    # def __process_author(self, text: str):
        # if self.last_chunk_label == Label.AUTHOR:
        #     self.authors[len(self.authors) - 1] += text
        # else:
        #     self.authors.append(text)

        # self.last_chunk_label = Label.AUTHOR