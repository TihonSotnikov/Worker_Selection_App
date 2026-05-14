import os
import asyncio
from typing import Literal
from logging import getLogger

import torch
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from faster_whisper import WhisperModel
from silero import silero_tts

from app.core.config import settings


LOGGER = getLogger(settings.LOGGER)
TESTING = os.getenv("TESTING", "0") == "1"

MODEL_NAME = "Qwen/Qwen3.5-4B"
QUANTIZATION_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)
MODEL_KWARGS = {
    "quantization_config": QUANTIZATION_CONFIG,
    "device_map": "cuda:0" if torch.cuda.is_available() else "cpu",
    "dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
}

gpu_lock = asyncio.Lock()

if TESTING:
    llm_pipeline = None
    llm_tokenizer = None
    llm_model = None
    gen_config = None
else:
    # llm_pipeline = pipeline(
    #     "text-generation",
    #     model=MODEL_NAME,
    #     model_kwargs=MODEL_KWARGS
    # )
    LOGGER.info(f'Running model on {MODEL_KWARGS["device_map"]}')
    llm_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    llm_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        **MODEL_KWARGS
    )
    gen_config = GenerationConfig(
        max_new_tokens = 4096,
        do_sample = False,
        temperature = 1.0,
        repetition_penalty = 1.0,
        pad_token_id = llm_tokenizer.eos_token_id
    )

transcriber = WhisperModel("large-v3-turbo")
tts_model, example_txt = silero_tts( # type: ignore
        language='ru',
        speaker="v5_5_ru"
    )
