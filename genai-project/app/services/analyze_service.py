import asyncio
import json
import logging
from typing import Optional

from fastapi import UploadFile
from sqlmodel import Session

from app.core.schemas import CandidateResult
from app.ai.extractor import extractor
from app.services.ai_service import save_upload_file, extract_from_file
from app.services.ml_service import ml_predict
from app.services.storage_service import save_candidate

logger = logging.getLogger(__name__)


async def process_candidate(
    upload_file: UploadFile,
    session: Session,
    model_ext: extractor,
    gpu_lock: Optional[asyncio.Lock] = None,
) -> CandidateResult:
    """
    Полный цикл обработки кандидата с использованием сервисного слоя.

    1. Сохранение файла (AI Service)
    2. Извлечение данных (AI Service)
    3. Предсказание удержания (ML Service)
    4. Запись в БД (Storage Service)

    Parameters
    ----------
    upload_file : UploadFile
        Файл резюме от рекрутера.
    session : Session
        Сессия БД.
    model_ext : extractor
        Инстанс модели экстрактора.
    gpu_lock : Optional[asyncio.Lock], default=None
        Блокировка для GPU.

    Returns
    -------
    CandidateResult
        Полный результат анализа (готов к отправке на frontend).
    """

    file_path = await save_upload_file(upload_file)

    try:
        full_name, raw_summary, vector = await extract_from_file(
            file_path, model_ext, gpu_lock
        )
    except Exception as e:
        logger.exception(f"AI Extraction failed for file {file_path}: {e}")
        raise

    try:
        retention_score, risk_factors = await ml_predict(vector)
    except Exception as e:
        logger.exception(f"ML Prediction failed for candidate {full_name}: {e}")
        raise

    db_candidate = save_candidate(
        session=session,
        full_name=full_name,
        raw_summary=raw_summary,
        retention_score=retention_score,
        risk_factors=risk_factors,
        vector=vector,
    )

    result = CandidateResult(
        id=str(db_candidate.id),
        full_name=db_candidate.full_name,
        raw_summary=db_candidate.raw_summary,
        vector=vector,
        retention_score=db_candidate.retention_score,
        risk_factors=json.loads(db_candidate.risk_factors),
    )

    return result
