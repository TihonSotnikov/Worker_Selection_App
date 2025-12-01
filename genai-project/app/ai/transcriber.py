import torch
import librosa
import numpy as np
from transformers import pipeline
from faster_whisper import WhisperModel
from datasets import load_dataset

# Убедитесь, что используете GPU, если есть (device=0), иначе уберите device
device = "cuda:0" if torch.cuda.is_available() else "cpu"
# device = "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32
# dtype = torch.float32

class transcriber:
    def __init__(self, model_name: str, *args, **kwargs):
        self._model = WhisperModel(model_name)
        # self._pipeline = pipeline(
        #     "automatic-speech-recognition",
        #     model=model_name,
        #     dtype=dtype,
        #     device=device,
        #     *args,
        #     **kwargs
        # )
    
    def __call__(self, *args, **kwds):
        return self._model.transcribe(*args, **kwds)
        # return self._pipeline(*args, **kwds)

    def change_model(self, model_name, *args, **kwargs):
        del self._model
        self._model = WhisperModel(model_name)

# "deepdml/faster-whisper-large-v3-turbo-ct2"
trans = transcriber("medium", language="ru")

filename = "test.wav"
# dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
# sample = dataset[0]["audio"]

segments, info = trans(
    filename, 
)

for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))