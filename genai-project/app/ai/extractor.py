import torch
from logging import getLogger

# import
import outlines
from time import time
from transformers import PreTrainedTokenizer, PreTrainedModel, BatchEncoding, GenerationConfig
# from lmformatenforcer import JsonSchemaParser
# from lmformatenforcer.integrations.transformers import (
#     build_transformers_prefix_allowed_tokens_fn,
# )
from colorama import init

from app.core.schemas import CandidateSummary
from app.core.config import settings
from app.ai.models import gen_config, gpu_lock
from outlines.inputs import Chat

init(autoreset=True)


SYSTEM_PROMPT_EXTRACT = """
Ты - HR-ассистент, помогающий с отбором кандидатов на работу.
Твоя задача - анализировать резюме и сопроводительные письма,
и на их основе составлять отчёт в формате json по модели CandidateSummary:
```
class ShiftPreference(IntEnum):
    DAY_ONLY = 0    # Только дневные смены
    NIGHT_ONLY = 1  # Только ночные смены
    ANY = 2         # Готов работать в любое время

class CandidateVector:
    skills_verified_count: int
    years_experience: float
    commute_time_minutes: int
    shift_preference: ShiftPreference 
    salary_expectation: int
    has_certifications: bool

class CandidateSummary:
    full_name: str    # ФИО кандидата
    raw_summary: str  # Краткое резюме от LLM (<150 символов)
    vector: CandidateVector
```

Если имя не указано, напиши "Не указано".
Резюме "raw_summary" должно быть кратким!
2-3 предложения, только важная информация.
Строки текста пиши на русском языке.
Не добавляй ничего лишнего, соблюдай синтаксис JSON.
"""

GEN_CFG = GenerationConfig(
    max_new_tokens = 2048,
    do_sample = False,
    temperature = 1.0,
    repetition_penalty = 1.0
)


class Extractor:
    """
    AI-модуль для извлечения структурированных данных кандидата из текстового резюме.

    Attributes
    ----------
    _pipeline : transformers.Pipeline
        Модель трансформера для генерации текста.
    _parser : JsonSchemaParser
        Парсер для обеспечения соответствия вывода заданной JSON-схеме.
    _prefix_func : callable
        Функция для ограничения токенов вывода в соответствии с JSON-схемой.

    Methods
    -------
    __call__(prompt, *args, **kwds)
        Вызывает модель трансформера с ограничением вывода по схеме.

    """

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, *args, **kwargs):
        logger = getLogger(settings.LOGGER)
        # logger.info(f"Initializing extractor with model {model_name}...")

        self._model = outlines.models.transformers.from_transformers(model, tokenizer)
        self._tokenizer = tokenizer
        self._generator = outlines.Generator(self._model, CandidateSummary)
        # logger.info(f"Extractor initialized on {device}.")
        self._logger = logger

    def __call__(self, prompt, *args, **kwds):
        tm = time()
        if self._logger:
            self._logger.info("Starting extraction process.")
        
        messages = Chat([
            {"role": "system", "content": SYSTEM_PROMPT_EXTRACT},
            {"role": "user", "content": prompt},
        ])
        
        output = self._generator(messages, max_new_tokens=2048) # type: ignore

        sum_json_str = str(output).strip("\n ")
        sum_json_str = sum_json_str.replace("\t", "  ").replace("\n", " ")
        if self._logger:
            self._logger.info("LLM output received.")

        try:
            candidate_summary = CandidateSummary.model_validate_json(sum_json_str)
            inference_time = time() - tm
            if self._logger:
                self._logger.info(
                    f"LLM output successfully parsed as CandidateSummary JSON. (Took {inference_time:.1f} sec.)"
                )
        except Exception as e:
            if self._logger:
                self._logger.error(
                    f"Error parsing LLM output as CandidateSummary JSON.\nString being parsed:\n{sum_json_str}"
                )
            raise e
        return (
            candidate_summary.full_name,
            candidate_summary.raw_summary,
            candidate_summary.vector,
        )


def get_vram_info(device_name: str):
    """
    Get total and free VRAM gigabytes by device name.

    Returns
    -------
    Tuple[float, float]
        Total and free VRAM value for `device_name`.
    """
    if not torch.cuda.is_available():
        return 0, 0
    device = torch.device(device_name)
    props = torch.cuda.get_device_properties(device)
    total_vram = props.total_memory / (1024**3)
    free_vram, _ = torch.cuda.mem_get_info(device)
    free_vram = free_vram / (1024**3)
    return total_vram, free_vram
