import torch
import json
import logging
from time import time
from transformers import pipeline
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
from app.core.schemas import CandidateVector, CandidateSummary
from colorama import init, Fore, Back, Style

init(autoreset=True)
logger = logging.getLogger("uvicorn")

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


class extractor:
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
    def __init__(self, model_name: str, *args, **kwargs):
        logger.info(f"Initializing extractor with model {model_name}...")
        self._pipeline = pipeline(
            "text-generation",
            model=model_name,
            device_map="auto",
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            *args,
            **kwargs
        )
        self._sum_parser = JsonSchemaParser(CandidateSummary.model_json_schema())
        self._sum_prefix_func = build_transformers_prefix_allowed_tokens_fn(self._pipeline.tokenizer, self._sum_parser)
        logger.info(f"Extractor initialized.")
    
    def __call__(self, prompt, *args, **kwds):
        tm = time()
        logger.info("Starting extraction process.")
        sum_json_str = self._pipeline(
            [[{"role": "system", "content": SYSTEM_PROMPT_EXTRACT}, {"role": "user", "content": prompt + ' /no_think'}]],
            *args,
            max_new_tokens=2048,
            prefix_allowed_tokens_fn=self._sum_prefix_func,
            repetition_penalty=1.15,
            return_full_text=False,
            **kwds
            )[-1]
        sum_json_str = str.strip(sum_json_str[0]['generated_text'], '\n ')
        sum_json_str = sum_json_str.replace('\t', '  ').replace('\n', ' ')
        logger.info("LLM output received.")
        
        try:
            candidate_summary = CandidateSummary.model_validate_json(sum_json_str)
            inference_time = time() - tm
            logger.info(f"LLM output successfully parsed as CandidateSummary JSON. (Took {inference_time:.1f} sec.)")
        except Exception as e:
            logger.error(f"Error parsing LLM output as CandidateSummary JSON.\nString being parsed:\n{sum_json_str}")
            raise e
        return candidate_summary.full_name, candidate_summary.raw_summary, candidate_summary.vector
