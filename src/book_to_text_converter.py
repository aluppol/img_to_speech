from img_to_text_converter import ImgToTextConverter

import re
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import spacy
from sentence_transformers import SentenceTransformer



class Chapter:
  def __init__(self, title: str, text: str, annotation: str):
    self.title = title
    self.text = text
    self.annotation = annotation


class BookToText:
  '''
    Initially I will work with pdf for simplicity using img_to_text converter to get raw text before making sense of it, multiple input formats can be added later.
  '''
  def convert(self, url: str) -> str:
    raw_text = ImgToTextConverter.convert(url)

    # Process Data
    df = self.preprocess_data(raw_text)
    embeddings = self.extract_text_embeddings(df)
    labels = ['title', 'main_text', 'annotation']  # Example labels; adjust based on training data
    clf = self.train_classifier(df, embeddings, labels)

    # Organize Text
    organized_text = self.organize_text(df, clf)

    return organized_text
    

  # Step 1: Feature Engineering
  def preprocess_data(self, data):
    df = pd.DataFrame(data)
    df['text_length'] = df['text'].apply(len)
    df['text_clean'] = df['text'].apply(lambda x: re.sub(r'[^\w\s]', '', x.lower()))
    return df

  # Step 2: NLP Feature Extraction (e.g., embeddings)
  def extract_text_embeddings(self, data, model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(data['text_clean'].tolist())
    return embeddings

  # Step 3: Classification Model
  def train_classifier(self, df, embeddings, labels):
    X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    return clf

  # Step 4: Text Grouping and Organization
  def organize_text(self,df, model):
    df['predicted_label'] = model.predict(self.extract_text_embeddings(df))
    grouped = df.groupby('page')
    chapters = []
    for page, group in grouped:
      chapter = {"title": "", "main_text": [], "annotations": []}
      for _, row in group.iterrows():
        if row['predicted_label'] == 'title':
          chapter['title'] = row['text']
        elif row['predicted_label'] == 'main_text':
          chapter['main_text'].append(row['text'])
        elif row['predicted_label'] == 'annotation':
          chapter['annotations'].append(row['text'])
      chapters.append(chapter)
    return chapters
