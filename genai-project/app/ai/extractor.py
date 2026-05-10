import torch
import logging
from time import time
from transformers import pipeline
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import (
    build_transformers_prefix_allowed_tokens_fn,
)
from app.core.schemas import CandidateSummary

SYSTEM_PROMPT_EXTRACT = """
Ты - HR-ассистент. Твоя задача - извлечь данные из резюме в JSON CandidateSummary:
- full_name: ФИО кандидата (если нет - "Не указано")
- raw_summary: кратко (2-3 предл., <150 симв.)
- vector:
    - skills_verified_count: кол-во подтвержденных навыков
    - years_experience: опыт работы в годах
    - age: возраст кандидата (число)
    - commute_time_minutes: время в пути в минутах (если нет - 45)
    - shift_preference: 0 (день), 1 (ночь), 2 (любой)
    - salary_expectation: зарплатные ожидания (число)
    - has_certifications: true/false (есть ли сертификаты/корочки)
    - education_level: 0 (среднее), 1 (спец), 2 (колледж), 3 (высшее)
    - previous_turnovers: количество прошлых мест работы/увольнений
    - family_status: 0 (нет), 1 (брак), 2 (дети), 3 (один родитель)
    - housing_type: 0 (свое), 1 (аренда), 2 (общага), 3 (родители)
    - has_transport: true (есть машина/права), false (нет)

Пиши на русском языке. Не добавляй ничего лишнего.
"""

class extractor:
    """
    AI-модуль для извлечения 12 структурированных признаков кандидата.
    """

    def __init__(self, model_name: str, logger: logging.Logger = None, *args, **kwargs):
        if logger:
            logger.info(f"Initializing extractor with model {model_name}...")
        self._pipeline = pipeline(
            "text-generation",
            model=model_name,
            device_map="auto",
            dtype="auto",
            *args,
            **kwargs,
        )
        self._sum_parser = JsonSchemaParser(CandidateSummary.model_json_schema())
        self._sum_prefix_func = build_transformers_prefix_allowed_tokens_fn(
            self._pipeline.tokenizer, self._sum_parser
        )
        self._logger = logger

    def __call__(self, prompt, *args, **kwds):
        tm = time()
        sum_json_str = self._pipeline(
            [
                [
                    {"role": "system", "content": SYSTEM_PROMPT_EXTRACT},
                    {"role": "user", "content": prompt + " /no_think"},
                ]
            ],
            *args,
            max_new_tokens=2048,
            prefix_allowed_tokens_fn=self._sum_prefix_func,
            repetition_penalty=1.15,
            return_full_text=False,
            **kwds,
        )[-1]
        
        sum_json_str = str.strip(sum_json_str[0]["generated_text"], "\n ")
        sum_json_str = sum_json_str.replace("\t", "  ").replace("\n", " ")

        try:
            candidate_summary = CandidateSummary.model_validate_json(sum_json_str)
            if self._logger:
                self._logger.info(f"Extracted data for: {candidate_summary.full_name}")
        except Exception as e:
            if self._logger:
                self._logger.error(f"JSON Parsing error: {e}")
            raise e
            
        return (
            candidate_summary.full_name,
            candidate_summary.raw_summary,
            candidate_summary.vector,
        )
