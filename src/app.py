from typing import List
import json
from pathlib import Path

from TextExtractor import TextExtractor
from TextClassifier import TextClassifier, TrainingData
from LabelTransformer import LabelTransformer
from TextAssembler import TextAssembler
from TextToSpeech import TextToSpeech, TextToSpeechPipeline


def generate_training_data(path: str):
  text_extractor = TextExtractor()
  label_transformer = LabelTransformer()

  model_path = 'src/models/book_text_classifier_model'
  text_classifier = TextClassifier(model_path)
    
  from_page, to_page = (102, 102)
  featured_text = text_extractor.extract('statics/roadto.pdf', from_page, to_page)
  text, num_features = text_classifier.preprocess_input(featured_text)
  labels = text_classifier.predict(text, num_features)
  for i in range(len(labels)):
    featured_text[i]['label'] = str(label_transformer.to_str(labels[i])).split('.')[1]

  safe_training_data(featured_text, path)

def train_text_classifier(training_data_dir: str, model_dir: str, epochs=5, loss_limit=0.5):
  text_classifier = TextClassifier(model_dir)
    
  text_classifier.train_model(training_data_dir, epochs=epochs, loss_limit=loss_limit)
  if not text_classifier.model_dir.exists():
    text_classifier.model_dir.mkdir()
  text_classifier.save_model(text_classifier.model_dir)

def safe_training_data(data: List[TrainingData], file_path: str):
  save_to_json(data, file_path)
  
def save_to_json(data, path: str):
  '''
  Save the given data to a JSON file at the specified path.

  :param data: The data to be saved (must be serializable to JSON).
  :param path: The file path where the JSON will be saved.
  '''
  with open(path, 'w', encoding='utf-8') as file:
    json.dump(data, file, ensure_ascii=False, indent=4)

def pdf_to_voice_pipeline(pdf_file_path: str, mp3_folder_path: str):
  text_extractor = TextExtractor()
  text_assembler = TextAssembler()

  model_path = Path('src/models/book_text_classifier_model')
  if not model_path.exists() or not any(model_path.iterdir()):
    model_path = 'aluppol/img_to_speech-book_text_classifier'
  text_classifier = TextClassifier(model_path)

  try:
    for featured_text_page in text_extractor.extract(pdf_file_path, 41, 43):
      classified_text_page = text_classifier.classify_featured_text(featured_text_page)
      chapters = []
      for chapter in text_assembler.process_classified_text(classified_text_page):
          chapters.append(chapter)
      text_assembler.save_chapter()
      chapter = text_assembler.chapter
      print(chapter.title)
      print(chapter.text)
          
  except Exception as e:
      print(f'Error: {e}')

def train_text_to_speech(model_path: str):
  text_to_speech = TextToSpeech(model_path)
  text_to_speech_pipeline = TextToSpeechPipeline()
  audio = text_to_speech.convert('The control of the production of wealth is the control ofhuman life itself.\n'
                         + 'Hilaire Bel/oc.\n Most planners who have seriously considered the practicalaspects of their task have little '
                         + 'doubt that a directed economymust be run on more or less dictatorial lines.  That the complexsystem of interrelated'
                         + ' activities, if it is to be consciously directedat all, must be directed by a single staff of experts, and '
                         + 'thatultimate responsibility and power must rest in the hands of acommander-in-chief, whose actions must not '
                         + 'be fettered bydemocratic procedure, is too obvious a consequence of under-lying ideas of central planning not '
                         + 'to command fairly generalassent.  The consolation our planners offer us is that this authori-tarian direction '
                         + 'will apply "only" to economic matters.  One ofthe most prominent American planners, Mr.  Stuart Chase, assuresus, '
                         + 'for instance, that in a planned society "political democracycan remain if it confines itself to all but economic matter". '
                         + ' Suchassurances are usually accompanied by the suggestion that bygiving up freedom in what are, or ought to be, '
                         + 'the less import-ant aspects of our lives, we shall obtain greater freedom in thepursuit of higher values.  '
                         + 'On this ground people who abhor theidea of a political dictatorship often clamour for a dictator inthe economic '
                         + 'field. The arguments used appeal to our best instincts and oftenattract the finest minds.  If planning really did '
                         + 'free us from theless important cares and so made it easier to render our existenceone of plain living and high '
                         + 'thinking, who would wish tobelittle such an ideal?  If our economic activities really concernedonly the inferior '
                         + 'or even more sordid sides of life, of course weought to endeavour by all means to find a way to relieve our-selves '
                         + 'from the excessive care for material ends, and, leavingthem to be cared for by some piece of utilitarian machinery, '
                         + 'setour minds free for the higher things of life. Unfortunately the assurance people derive from this beliefthat the '
                         + 'power which is exercised over economic life is a powerover matters of secondary importance only, and which makesthem '
                         + 'take lightly the threat to the freedom of our economicpursuits, is altogether unwarranted.  It is largely a consequence '
                         + 'ofthe erroneous belief that there are purely economic ends separ-ate from the other ends of life.  Yet, apart from the '
                         + 'pathologicalcase of the miser, there is no such thing.  The ultimate ends of theactivities ofreasonable beings are '
                         + 'never economic.  Strictly speak-ing there is no "economic motive" but only economic factorsconditioning our striving '
                         + 'for other ends.  What in ordinarylanguageismisleadinglycalledthe"economicmotive"means merely the desire for general '
                         + 'opportunity, the desire forpower to achieve unspecified ends.  Cf. L. Robbins, The Economic Causes of War, 1939, '
                         + 'Appendix.  If we strive for money it isbecause it offers us the widest choice in enjoying the fruits of')
  text_to_speech_pipeline.save_waveform_as_audio(audio, 'statics/output_audio/test1.wav')
  
if __name__ == '__main__':
  # pdf_to_voice_pipeline('statics/roadto.pdf')
  # generate_training_data('statics/model_training_data/roadto/change_name.json')
  # train_text_classifier('statics/model_training_data/roadto', 'src/models/img_to_speech-book_text_classifier', loss_limit=4)
  train_text_to_speech('src/models/img_to_speech-text_to_speech_model')