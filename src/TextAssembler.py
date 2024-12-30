from typing import List, Generator
import re

from TextCategorizer import ClassifiedText
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
        annotation_references: List[str] = None,
    ):
        self.text = text
        self.annotation_references = annotation_references if annotation_references else []

class Annotation:
    def __init__(
        self,
        text: str = '',
        reference: str = None,
    ):
        self.text = text
        self.reference = reference

class TextAssembler:
    def __init__(self):
        self.title = None
        self.chapter_title = None
        self.sentenses: List[Sentense] = [Sentense()]
        self.annotations: List[Annotation] = []
        self.annotation_references = []
        self.epigraphs = []
        self.authors = []
        self.last_chunk_label: int = None
        self.classification_mistakes = []
        self.chapter=None

    def process_classified_text(self, classified_text: List[ClassifiedText]) -> Generator[Chapter, None, None]:
        for chunk in classified_text:
            match chunk.label:
                case Label.TITLE.value:
                    self.__process_title(chunk.text)
                
                case Label.CHAPTER_TITLE.value:
                    self.__process_chapter_title(chunk.text)

                case Label.CHAPTER_TEXT.value:
                    self.__process_chapter_text(chunk.text)

                case Label.ANNOTATION.value:
                    self.__process_annotation(chunk.text)
                
                case Label.ANNOTATION_REFERENCE.value:
                    self.__process_annotation_references(chunk.text)

                case Label.EPIGRAPH.value:
                    self.__process_epigraph(chunk.text)

                case Label.AUTHOR.value:
                    self.__process_author(chunk.text)

                case _:
                    pass    # add to logging all passed chunks
        
        if self.chapter:
            chapter = self.chapter
            self.chapter = None
            yield chapter

    def save_chapter(self):
        chapter_text = ''
        for i in range(len(self.epigraphs)):
            chapter_text += self.epigraphs[i] + '\n'
            chapter_text += self.authors[i] + '\n'

        for sentense in self.sentenses:
            chapter_text += ' ' + sentense.text
            for reference in sentense.annotation_references:
                annotation = next((ann for ann in self.annotations if ann.reference == reference), None)
                if annotation:
                    chapter_text += ' ' + annotation.text
                else:
                    raise Exception(f'No reference for "{reference}" annotation found!')
                        
        self.chapter = Chapter(self.chapter_title, chapter_text)

        self.chapter_title = None
        self.annotation_references = []
        self.annotations = []
        self.epigraphs = []
        self.authors = []
        self.sentenses = []
        self.last_chunk_label = None

    def __process_title(self, text: str):
        if (self.title or not self.last_chunk_label) and self.last_chunk_label == Label.TITLE:
            self.title += text
            self.last_chunk_label = Label.TITLE
        elif self.last_chunk_label != Label.TITLE:
            self.classification_mistakes.append(ClassifiedText(Label.TITLE, text))

    def __process_chapter_title(self, text: str):
        if self.last_chunk_label != Label.CHAPTER_TITLE:
            if self.chapter_title:
                self.save_chapter()
            self.chapter_title = text
            self.last_chunk_label = Label.CHAPTER_TITLE

        else:
            self.chapter_title += text


    def __process_chapter_text(self, text: str):
        if self.last_chunk_label == Label.ANNOTATION_REFERENCE:
            ref = self.annotation_references[len(self.annotation_references) - 1]
            current_sentense_text = self.sentenses[len(self.sentenses) - 1].text.strip()
            self.sentenses[(len(self.sentenses) - 1) if current_sentense_text else (len(self.sentenses) - 2)].annotation_references.append(ref)

        end_of_sentense_match = re.match(r"(.*?[.!?]\s*)(.*)", text, re.DOTALL)
        while end_of_sentense_match:
            self.sentenses[len(self.sentenses) - 1].text += end_of_sentense_match.group(1)
            self.sentenses.append(Sentense())
            text = end_of_sentense_match.group(2)
            end_of_sentense_match = re.match(r"(.*?[.!?]\s*)(.*)", text, re.DOTALL)

        self.sentenses[len(self.sentenses) - 1].text += text
        self.last_chunk_label = Label.CHAPTER_TEXT

    def __process_annotation(self, text: str):
        if self.last_chunk_label == Label.ANNOTATION_REFERENCE:
            self.annotations.append(Annotation(text, self.annotation_references[len(self.annotation_references) - 1]))
            self.annotation_references.append('')
        else:
            self.annotations[len(self.annotations) - 1].text += text

        self.last_chunk_label = Label.ANNOTATION

    def __process_annotation_references(self, text: str):
        if self.last_chunk_label == Label.ANNOTATION_REFERENCE:
            self.annotation_references[len(self.annotation_references) - 1] += re.sub(r"[^a-zA-Z0-9]", "", text)
        else:
            self.annotation_references.append(re.sub(r"[^a-zA-Z0-9]", "", text))
            self.last_chunk_label = Label.ANNOTATION_REFERENCE

    def __process_epigraph(self, text: str):
        if self.last_chunk_label == Label.EPIGRAPH:
            self.epigraphs[len(self.epigraphs) - 1] += text
        else:
            self.epigraphs.append(text)
        
        self.last_chunk_label = Label.EPIGRAPH

    def __process_author(self, text: str):
        if self.last_chunk_label == Label.AUTHOR:
            self.authors[len(self.authors) - 1] += text
        else:
            self.authors.append(text)

        self.last_chunk_label = Label.AUTHOR