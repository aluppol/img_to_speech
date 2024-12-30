# Img-to-Speech: Convert Books into Audiobooks

**Img-to-Speech** is a project designed to streamline the conversion of books into audiobooks by combining advanced text extraction, classification, and text-to-speech technologies.

---

## Features

1. **PDF Text Extraction**:
   - Uses `pymupdf` to extract text from PDF files.

2. **Text Classification**:
   - Employs a custom BERT-based classification model to organize extracted text into meaningful categories:
     - **HEADER**
     - **FOOTER**
     - **ANNOTATION**
     - **TITLE**
     - **CHAPTER_TITLE**
     - **CHAPTER_TEXT**
     - **OTHER**
     - **ANNOTATION_REFERENCE**
     - **EPIGRAPH**
     - **AUTHOR**
   - The classification model is hosted on [Hugging Face Hub](https://huggingface.co/aluppol/img_to_speech-book_text_classifier).

3. **Text Structuring**:
   - Assembles classified text into structured chapters for audiobook generation.

4. **Text-to-Speech (TTS)**:
   - Planned integration of a TTS model for converting structured chapters into high-quality audio files.
   - The TTS model will also be hosted on Hugging Face Hub upon development.

---

## Future Plans

- Implementing and training a TTS model.
- Ensuring seamless integration between text structuring and audio generation.
- Supporting additional formats and languages.

---

## Installation and Usage

### Dependencies

- Python 3.8+
- `pymupdf` for PDF text extraction
- `transformers` for BERT-based classification
- `torch` for deep learning workflows
- Additional dependencies listed in `requirements.txt`.

### Steps to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/img_to_speech.git
   cd img_to_speech

2. Install dependencies:
   ```bash
   pip install -r requirements.txt

3. Install sox:
   ```bash
   sudo apt-get install sox libsox-dev libsox-fmt-all
   sudo apt-get install libsndfile1

4. Install tesseract (for text from image extraction):
   ```bash
   sudo apt-get install tesseract-ocr




## Licensing

This project is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)**:

- **Personal use**: Allowed for everyone.
- **Commercial use**: Prohibited unless explicitly approved by the author.

You may copy, distribute, remix, transform, and build upon this work non-commercially, as long as you credit the author. Commercial use is not permitted without prior approval.

For more details, refer to the [CC BY-NC 4.0 License](https://creativecommons.org/licenses/by-nc/4.0/).

## Contributing

Contributions are welcome! Please submit a pull request or open an issue to suggest improvements.

## Contact

For questions or commercial inquiries, please contact:  
albert.y.luppol@gmial.com

## Author

This project is maintained by Albert Luppol.

Feel free to reach out with any questions, suggestions, or contributions.

Copyright (c) 2024 Albert Luppol


Usefull dev commands:
```bash
pdftk statics/roadto.pdf cat 9 output statics/roadto_9.pdf