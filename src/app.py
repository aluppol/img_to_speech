from TextExtractor import TextExtractor
from TextClassifier import TextClassifierPipeline
from LabelTransformer import Label

import pandas as pd
import json

def run():
    text_extractor = TextExtractor()
    # text_classifier = BookToText()
    try:
      # result = converter.convert("test/statics/c.jpg")
      # print("Extracted Text:\n", result)
      # result = converter.convert("test/statics/roadto_p1.pdf")
      # print("Extracted Text:\n", result)
      # result = book_converter.convert("test/statics/roadto_p1-10.pdf")

      model_path = "text_classifier_model.pth"
      bert_model_name = "bert-base-uncased"
      num_numeric_features = 7 # 
      num_classes = 8 # Header, Footer, Annotation, Chapter Title, Chapter Text, Annotation, Other, Annotation Reference

      pipeline = TextClassifierPipeline(model_path, bert_model_name, num_numeric_features, num_classes)
      
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
  run()

  
  # Example data
  # text_data = [
  #     "The product is amazing and exceeded my expectations!",
  #     "It arrived late, and the quality was terrible.",
  #     "It's okay. Nothing special, but not bad either."
  # ]
  # numeric_features = [
  #     [50, 0.9, 2],  # [text_length, sentiment_score, page_number]
  #     [47, -0.7, 3],
  #     [44, 0.0, 1]
  # ]
  # labels = [0, 1, 2]  # Example class labels: 0 -> Positive, 1 -> Negative, 2 -> Neutral

  # # Train the model
  # pipeline.train(text_data, numeric_features, labels, epochs=3)

  # # Predict
  # predictions = pipeline.predict(text_data, numeric_features)
  # class_names = ["Positive", "Negative", "Neutral"]
  # final_predictions = [class_names[label] for label in predictions]
  # print("Predictions:", final_predictions)

  # # Save the model
  # pipeline.save_model()