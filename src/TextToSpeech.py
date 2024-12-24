from transformers import VitsTokenizer, VitsModel, set_seed, Trainer, TrainingArguments
import torch
from pathlib import Path
import torchaudio


class TextToSpeech:
    def __init__(self, model_path: str):
        self.__initiate_model(model_path)
        set_seed(613)
        
    def convert(self, text: str) -> torch.Tensor:
        """Convert the given text to speech"""
        inputs = self.tokenizer(text=text, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(inputs['input_ids'])
        return outputs.waveform

    def save_model(self, save_dir: Path):
        if not save_dir.exists():
            save_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)

    def __load_model(self, model_source: str):
        self.tokenizer = VitsTokenizer.from_pretrained(model_source)
        self.model = VitsModel.from_pretrained(model_source)
    
    def __initiate_model(self, model_path: int = None, initial_model_repo='facebook/mms-tts-eng'):
        if model_path and Path(model_path).exists():
            self.__load_model(model_path)
        else:
            self.__load_model(initial_model_repo)


class TextToSpeechPipeline():
    def __init__(self):
        pass

    def save_waveform_as_audio(self, waveform: torch.Tensor, file_path: str):
        """
        Saves a PyTorch tensor (waveform) to an audio file.
        
        Args:
            waveform (torch.Tensor): The waveform to save.
            file_path (str): The path where to save the audio file.
        """

        # Save as a .wav file using torchaudio
        torchaudio.save(file_path, waveform, 22050, format='wav')  # 22050 is a common sample rate - the quality of audio treck