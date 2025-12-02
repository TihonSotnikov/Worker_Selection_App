import torch
import json
from transformers import pipeline
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
from app.core.schemas import CandidateVector, CandidateSummary

SYSTEM_PROMPT_EXTRACT = """
Ты - HR-ассистент, помогающий с отбором кандидатов на работу.
Твоя задача - анализировать резюме и сопроводительные письма,
и на их основе составлять отчёт в формате json следующего вида:
{schema}.
""".format(schema=json.dumps(CandidateVector.model_json_schema(), indent=2, ensure_ascii=False))

SYSTEM_PROMPT_SUMMARY = """
Ты - HR-ассистент, помогающий с отбором кандидатов на работу.
Твоя задача - написать полное имя и краткое резюме кандидата
в следующем json-формате:
{schema}.

Если имя не указано, напиши "Не указано".
Не добавливай ничего лишнего, пиши только по существу.
""".format(schema=json.dumps(CandidateSummary.model_json_schema(), indent=2, ensure_ascii=False))


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
        self._pipeline = pipeline(
            "text-generation",
            model=model_name,
            device_map="auto",
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            *args,
            **kwargs
        )
        self._vec_parser = JsonSchemaParser(CandidateVector.model_json_schema())
        self._vec_prefix_func = build_transformers_prefix_allowed_tokens_fn(self._pipeline.tokenizer, self._vec_parser)
        self._sum_parser = JsonSchemaParser(CandidateSummary.model_json_schema())
        self._sum_prefix_func = build_transformers_prefix_allowed_tokens_fn(self._pipeline.tokenizer, self._sum_parser)
    
    def __call__(self, prompt, *args, **kwds):
        vec_json_str = self._pipeline(
            [[{"role": "system", "content": SYSTEM_PROMPT_EXTRACT}, {"role": "user", "content": prompt}]],
            *args,
            prefix_allowed_tokens_fn=self._vec_prefix_func,
            return_full_text=False,
            **kwds
            )[-1]
        vec_json_str = str.strip(vec_json_str[0]['generated_text'], '\n ')
        candidate_vector = CandidateVector.model_validate_json(vec_json_str)
        summary_str = self._pipeline(
            [[{"role": "system", "content": SYSTEM_PROMPT_SUMMARY}, {"role": "user", "content": prompt}]],
            *args,
            prefix_allowed_tokens_fn=self._sum_prefix_func,
            return_full_text=False,
            **kwds
            )[-1][0]['generated_text']
        summary_str = str.strip(summary_str, '\n ')
        candidate_summary = CandidateSummary.model_validate_json(summary_str)
        return candidate_vector, candidate_summary.full_name, candidate_summary.raw_summary