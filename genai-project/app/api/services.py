import asyncio
from pathlib import Path
from typing import Tuple
import uuid
import json

from fastapi import UploadFile
from fastapi.concurrency import run_in_threadpool
from sqlmodel import Session

from app.core.config import settings
from app.core.schemas import CandidateVector, CandidateResult
from app.core.enums import ShiftPreference
from app.api.models_db import CandidateTable
from app.ai.transcriber import transcriber
from app.ai.extractor import extractor
from app.ai.extractor import SYSTEM_PROMPT_EXTRACT


async def save_upload_file(upload_file: UploadFile) -> Path:
    """
    Сохраняет загруженный файл на диск с уникальным именем.

    Parameters
    ----------
    upload_file : UploadFile
        Объект файла от FastAPI. Содержит:
        - filename: оригинальное имя файла
        - file: поток байтов (file-like object)

    Returns
    -------
    Path
        Полный путь к сохранённому файлу.

    Notes
    -----
    Функция асинхронная для оптимизации IO операций FastAPI через event loop.
    """

    file_extension = Path(upload_file.filename).suffix
    unique_filename = f"{uuid.uuid4()}{file_extension}"

    file_path = Path(settings.UPLOAD_DIR) / unique_filename
    file_path.parent.mkdir(parents=True, exist_ok=True)

    content = await upload_file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    return file_path


async def ai_extract(file_path: Path, ext: extractor, gpu_lock: asyncio.Lock = None) -> Tuple[str, str, CandidateVector]:
    """AI экстракция данных из резюме."""

    with open(file_path, "r", encoding="utf-8") as f:
        resume_text = f.read()

    if gpu_lock:
        async with gpu_lock:
            name, summary, vector = await run_in_threadpool(ext, resume_text)
    else:
        name, summary, vector = await run_in_threadpool(ext, resume_text)

    return name, summary, vector


async def ml_predict(vector: CandidateVector) -> Tuple[float, list[str]]:
    """Заглушка для модуля ML (предсказание удержания)."""
    score = 0.85
    risks = []

    if vector.commute_time_minutes > 60:
        score -= 0.3
        risks.append("Долгая дорога до работы (>60 мин)")

    if not vector.has_certifications:
        score -= 0.1
        risks.append("Отсутствуют сертификаты")

    if vector.years_experience < 2.0:
        score -= 0.2
        risks.append("Недостаточный опыт (<2 лет)")

    score = max(0.0, min(1.0, score))
    return score, risks


# async def ml_predict(vector: CandidateVector) -> Tuple[float, list[str]]:
#     """Вызов ML-модуля для предсказания удержания кандидата."""

#     try:
#         # Импорт и инициализация ML-модуля
#         from app.ml.predictor import RetentionPredictor
#         predictor = RetentionPredictor()

#         # Загрузка или обучение модели
#         try:
#             predictor.load_model("app/ml/model.pkl")
#         except FileNotFoundError:
#             predictor.train_model()
#             predictor.save_model("app/ml/model.pkl")

#         # Преобразование данных для модели
#         features = {
#             'skills_verified_count': vector.skills_verified_count,
#             'years_experience': vector.years_experience,
#             'commute_time_minutes': vector.commute_time_minutes,
#             'shift_preference': vector.shift_preference.value,
#             'salary_expectation': vector.salary_expectation,
#             'has_certifications': vector.has_certifications
#         }

#         # Выполнение предсказания
#         prediction = predictor.predict_retention(features)
#         risk_factors = predictor.explain_prediction(features)

#         return float(prediction['retention_probability']), risk_factors

#     except Exception as e:
#         # Fallback на детерминированные правила при ошибке
#         print(f"ML модуль недоступен: {e}")

#         # Базовые правила из ТЗ
#         score = 0.85
#         risks = []

#         if vector.commute_time_minutes > 60:
#             score -= 0.3
#             risks.append("Долгая дорога до работы (>60 мин)")

#         if not vector.has_certifications:
#             score -= 0.1
#             risks.append("Отсутствуют сертификаты")


#         if vector.years_experience < 2.0:
#             score -= 0.2
#             risks.append("Недостаточный опыт (<2 лет)")

#         score = max(0.0, min(1.0, score))
#         return score, risks


async def process_candidate(
    upload_file: UploadFile,
    session: Session,
    model_ext: extractor,
    gpu_lock: asyncio.Lock = None
) -> CandidateResult:
    """
    Полный цикл обработки кандидата.

    1. Сохранение файла
    2. Извлечение данных (AI)
    3. Предсказание удержания (ML)
    4. Запись в БД
    5. Формирование ответа

    Parameters
    ----------
    upload_file : UploadFile
        Файл резюме от рекрутера.
    session : Session
        Сессия БД (приходит из dependency injection в routes.py).

    Returns
    -------
    CandidateResult
        Полный результат анализа (готов к отправке на frontend).

    Raises
    ------
    Exception
        Любые ошибки обрабатываются в routes.py (там будет try/except).
    """

    file_path = await save_upload_file(upload_file)

    full_name, raw_summary, vector = await ai_extract(file_path, model_ext, gpu_lock)

    retention_score, risk_factors = await ml_predict(vector)

    db_candidate = CandidateTable(
        full_name=full_name,
        raw_summary=raw_summary,
        retention_score=retention_score,
        risk_factors=json.dumps(risk_factors, ensure_ascii=False),
        vec_skills_count=vector.skills_verified_count,
        vec_years_experience=vector.years_experience,
        vec_commute_minutes=vector.commute_time_minutes,
        vec_shift_preference=int(vector.shift_preference),
        vec_salary_expectation=vector.salary_expectation,
        vec_has_certifications=vector.has_certifications
    )

    session.add(db_candidate)
    session.commit()
    session.refresh(db_candidate)

    result = CandidateResult(
        id=str(db_candidate.id),
        full_name=db_candidate.full_name,
        raw_summary=db_candidate.raw_summary,
        vector=vector,
        retention_score=db_candidate.retention_score,
        risk_factors=json.loads(db_candidate.risk_factors)
    )

    return result


def get_all_candidates(session: Session) -> list[CandidateResult]:
    """
    Получает список всех кандидатов из БД.

    Используется для эндпоинта GET /history.

    Parameters
    ----------
    session : Session
        Сессия БД.

    Returns
    -------
    List[CandidateResult]
        Список всех кандидатов, отсортированных по дате (новые первые).
    """

    candidates = session.query(CandidateTable).order_by(
        CandidateTable.created_at.desc()
    ).all()

    results = []
    for db_candidate in candidates:
        vector = CandidateVector(
            skills_verified_count=db_candidate.vec_skills_count,
            years_experience=db_candidate.vec_years_experience,
            commute_time_minutes=db_candidate.vec_commute_minutes,
            shift_preference=ShiftPreference(db_candidate.vec_shift_preference),
            salary_expectation=db_candidate.vec_salary_expectation,
            has_certifications=db_candidate.vec_has_certifications
        )

        result = CandidateResult(
            id=str(db_candidate.id),
            full_name=db_candidate.full_name,
            raw_summary=db_candidate.raw_summary,
            vector=vector,
            retention_score=db_candidate.retention_score,
            risk_factors=json.loads(db_candidate.risk_factors)
        )
        results.append(result)

    return results
