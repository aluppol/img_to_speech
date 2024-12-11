from TextExtractor import TextExtractorPipeline
from TextClassifier import TextClassifierPipeline
from LabelTransformer import Label

def run():
    text_extractor = TextExtractorPipeline()

    model_path = "src/text_classifier_model.pth"
    bert_model_name = "bert-base-uncased"
    num_numeric_features = 7 # 
    num_classes = len(Label) + 1
    text_classifier = TextClassifierPipeline(model_path, bert_model_name, num_numeric_features, num_classes)

    try:
      pass # file processing logic
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
  run()
