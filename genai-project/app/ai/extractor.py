import torch
import json
from transformers import pipeline
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
from app.core.schemas import CandidateVector

SYSTEM_PROMPT = """
Ты - HR-ассистент, помогающий с отбором кандидатов на работу.
Твоя задача - анализировать резюме и сопроводительные письма,
и на их основе составлять отчёт в формате json следующего вида:
{schema}.
""".format(schema=json.dumps(CandidateVector.model_json_schema(), indent=2, ensure_ascii=False))


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
    __call__(*args, **kwds)
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
        self._parser = JsonSchemaParser(CandidateVector.model_json_schema())
        self._prefix_func = build_transformers_prefix_allowed_tokens_fn(self._pipeline.tokenizer, self._parser)
    
    def __call__(self, prompt, *args, **kwds):
        result = self._pipeline(
            [[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]],
            *args,
            prefix_allowed_tokens_fn=self._prefix_func,
            return_full_text=False,
            **kwds)[-1]
        json_output = str.strip(result[0]['generated_text'], '\n ')
        return CandidateVector.model_validate_json(json_output)