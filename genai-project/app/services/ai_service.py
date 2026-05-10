import asyncio
import uuid
from pathlib import Path
from typing import Tuple, Optional

from fastapi import UploadFile, HTTPException
from fastapi.concurrency import run_in_threadpool

from app.core.config import settings
from app.core.schemas import CandidateVector
from app.ai.extractor import extractor
from app.ai.transcriber import transcriber


async def save_upload_file(upload_file: UploadFile) -> Path:
    """
    Сохраняет загруженный файл на диск с уникальным именем.

    Parameters
    ----------
    upload_file : UploadFile
        Объект файла от FastAPI. Содержит:
        - filename: оригинальное имя файла
        - file: поток байтов

    Returns
    -------
    Path
        Полный путь к сохранённому файлу.
    """

    file_extension = Path(upload_file.filename).suffix
    unique_filename = f"{uuid.uuid4()}{file_extension}"

    file_path = Path(settings.UPLOAD_DIR) / unique_filename
    file_path.parent.mkdir(parents=True, exist_ok=True)

    content = await upload_file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    return file_path


async def extract_from_file(
    file_path: Path, ext: extractor, gpu_lock: Optional[asyncio.Lock] = None
) -> Tuple[str, str, CandidateVector]:
    """
    AI экстракция данных из резюме (текст или аудио).

    Parameters
    ----------
    file_path : Path
        Путь к сохраненному файлу резюме.
    ext : extractor
        Инстанс модели экстрактора.
    gpu_lock : Optional[asyncio.Lock], default=None
        Асинхронный лок для предотвращения OOM при работе с GPU.

    Returns
    -------
    Tuple[str, str, CandidateVector]
        Имя кандидата, краткое резюме и извлеченный вектор признаков.
    """

    extension = file_path.suffix.lower()

    if extension in [".wav", ".mp3"]:
        try:
            stt_model = transcriber("medium")
            segments, info = stt_model(str(file_path))
            resume_text = " ".join([segment.text for segment in segments])
        except Exception:
            raise HTTPException(
                status_code=400, detail="Ошибка при обработке аудиофайла"
            )
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            resume_text = f.read()

    if gpu_lock:
        async with gpu_lock:
            name, summary, vector = await run_in_threadpool(ext, resume_text)
    else:
        name, summary, vector = await run_in_threadpool(ext, resume_text)

    return name, summary, vector
