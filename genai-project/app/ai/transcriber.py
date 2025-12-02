import torch
<<<<<<< HEAD
<<<<<<< HEAD
import librosa
import numpy as np
from transformers import pipeline
from faster_whisper import WhisperModel
from datasets import load_dataset
=======
import numpy as np
from transformers import pipeline
from faster_whisper import WhisperModel
>>>>>>> 700d7a42a7cd7765a4e036b8432f8f677af0d13c
=======
import numpy as np
from transformers import pipeline
from faster_whisper import WhisperModel
>>>>>>> 2aa28140476ea4e90b169bbb282e9cc45e9a3374

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

# "deepdml/faster-whisper-large-v3-turbo-ct2"
<<<<<<< HEAD
<<<<<<< HEAD
trans = transcriber("medium", language="ru")

filename = "test.wav"
# dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
# sample = dataset[0]["audio"]

segments, info = trans(
    filename, 
)

for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
=======
# "medium"
# trans = transcriber("deepdml/faster-whisper-large-v3-turbo-ct2", language="ru")

# filename = "test.wav"
# dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
# sample = dataset[0]["audio"]

# for i in range(1, 4):
#     print(f"Iter {i}:", i)

#     segments, info = trans(
#         filename, 
#     )

#     for segment in segments:
#         print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
>>>>>>> 700d7a42a7cd7765a4e036b8432f8f677af0d13c
=======
# "medium"
>>>>>>> 2aa28140476ea4e90b169bbb282e9cc45e9a3374
