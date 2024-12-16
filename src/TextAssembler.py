from typing import List, Generator

from TextClassifier import ClassifiedText
from LabelTransformer import Label


class Chapter:
  def __init__(
        self,
        title: str,
        text: str,
    ):
    self.title = title
    self.text = text

class Sentense:
    def __init__(
        self,
        text: str = '',
        annotation: int = None,
    ):
        self.text = text
        self.annotation = annotation

class TextAssembler:
    def __init__(self):
        self.title = None
        self.chapter_title = None
        self.sentenses: List[Sentense] = []
        self.sentense = ''
        self.annotations = []
        self.annotation_references = []
        self.epigraphs = []
        self.authors = []
        self.last_chunk_label: int = None
        self.classification_mistakes = []
        self.chapter=None

    def process_classified_text(self, classified_text: List[ClassifiedText]) -> Generator[Chapter]:
        for chunk in classified_text:
            match chunk.label:
                case Label.TITLE:
                    self.__process_title(chunk.text)
                
                case Label.CHAPTER_TITLE:
                    self.__process_chapter_title(chunk.text)

                case Label.CHAPTER_TEXT:
                    self.__process_chapter_text(chunk.text)

                case Label.ANNOTATION:
                    self.__process_annotation(chunk.text)
                
                case Label.ANNOTATION_REFERENCE:
                    self.__process_annotation_reference(chunk.text)

                case Label.EPIGRAPH:
                    self.__process_epigraph(chunk.text)

                case Label.AUTHOR:
                    self.__process_author(chunk.text)

                case _:
                    raise NotImplementedError(f'"{chunk.label}" label is not implemented!')
        
        if self.chapter:
            chapter = self.chapter
            self.chapter = None
            yield chapter

    def __process_title(self, text: str):
        if (self.title or not self.last_chunk_label) and self.last_chunk_label == Label.TITLE:
            self.title += text
            self.last_chunk_label = Label.TITLE
        elif self.last_chunk_label != Label.TITLE:
            self.classification_mistakes.append(ClassifiedText(Label.TITLE, text))

    def __process_chapter_title(self, text: str):
        if self.chapter_title and self.last_chunk_label != Label.CHAPTER_TITLE:
            self.__save_chapter()
            self.chapter_title = text
            self.last_chunk_label = Label.CHAPTER_TITLE

        elif self.last_chunk_label == Label.CHAPTER_TITLE:
            self.chapter_title += text

    def __save_chapter(self):
        chapter_text = ''

        for i in range(len(self.epigraphs)):
            chapter_text += self.epigraphs[i] + '\n'
            chapter_text += self.authors[i] + '\n'

        for sentense in self.sentenses:
            chapter_text += ' ' + sentense.text
            if sentense.annotation:
                for i in range(len(self.annotation_references)):
                    if sentense.annotation == self.annotation_references[i]:
                        chapter_text += ' ' + self.annotations[i]
                        break
                        
        self.chapter = Chapter(self.chapter_title, chapter_text)

        self.chapter_title = None
        self.annotation_references = []
        self.annotations = []
        self.epigraphs = []
        self.authors = []
        self.sentenses = []
        self.last_chunk_label = None

    def __process_chapter_text(self, text: str):
        pass
