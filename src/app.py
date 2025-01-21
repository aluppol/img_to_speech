from PdfToWavConverter import PdfToWavConverter
from TrainingDataGenerator import TrainingDataGenerator
from TextClassifierTrainer import TextClassifierTrainer


if __name__ == '__main__':
  # training_data_generator = TrainingDataGenerator()
  # training_data_generator.generate_training_data('roadto', 225)

  # text_classifier_trainer = TextClassifierTrainer('statics/model_training_data/roadto')
  # text_classifier_trainer.train_text_classifier(loss_limit=1)

  pdf_to_wav_converter = PdfToWavConverter()
  pdf_to_wav_converter.convert_pdf_to_wav('statics/books/roadto.pdf', 'statics/output_audio')

  # pdf_to_voice_pipeline('statics/roadto.pdf', 'statics/output_audio')
  # pdf_to_wav_converter.generate_training_data('statics/books/roadto_9.pdf', 'statics/model_training_data/roadto/01-page-9-introduction')
  # train_text_classifier('statics/model_training_data/roadto', 'src/models/img_to_speech-book_text_classifier', loss_limit=4)
  # train_text_to_speech('src/models/img_to_speech-text_to_speech_model')
  # convert_pdf_to_wav('statics/books/roadto_9.pdf', 'statics/output_audio')
  pass