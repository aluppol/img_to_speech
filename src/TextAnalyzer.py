from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path


class TextAnalyzer:
    def __init__(self, model_dir='src/models/img_to_speech-book_text_analyzer_model', model_name='meta-llama/Llama-2-7b-chat-hf'):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
            self.model = AutoModelForCausalLM.from_pretrained(model_dir)
            print(f'Model ... loaded from {model_dir}')
        except Exception as e:
            print(f'Failed to load model from {model_dir} ({e}); Initializing model ... from {model_name}')
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)

            path_to_model_dir = Path(model_dir)
            if not path_to_model_dir.exists():
                Path(model_dir).mkdir(parents=True, exist_ok=True)

            self.tokenizer.save_pretrained(model_dir)
            self.model.save_pretrained(model_dir)
            

    def is_the_same_paragraph(self, end_of_page: str, beginning_of_next_page: str) -> bool:
        prompt = f'''Does sentences in the triple brackets belong to the same paragraph? \n
            """ {end_of_page}"""\n"""{beginning_of_next_page}"""\n
            Answer with "yes" or "no" only.'''
        inputs = self.tokenizer(prompt, return_tensors='pt')
        outputs = self.model.generate(**inputs, max_length=1000)
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return 'yes' in answer.lower()

    def split_text_to_paragraphs(self, text: str) -> List[str]:
        prompt = f'''Split text provided in the triple brackets into paragraphs: \n
            """ {text}"""\n
            Provide answer in the next form:
            Paragraph 1\n\nParagraph 2\n\nParagraph3\n\n...
            \n\n\nNo metter what not include anything else other then exactly the same text that was provided splitted into described format'''
        inputs = self.tokenizer(prompt, return_tensors='pt')
        outputs = self.model.generate(**inputs, max_length=1000)
        paragraphs = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return paragraphs.split("\n\n")

    def is_annotation_belong_to_the_text(self, text: str, annotation: str) -> bool:
        prompt = f'''Does annotation provided in the triple brackets belong to the text provided in double brackets? \n
            """ {annotation}"""\n""{text}""\n
            Answer with "yes" or "no" only.'''
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_length=1000)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)