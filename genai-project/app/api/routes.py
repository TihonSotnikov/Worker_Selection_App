from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, status, Request
from sqlmodel import Session
from typing import List

from app.api.database import get_session
from app.api.services import process_candidate, get_all_candidates
from app.core.schemas import CandidateResult

# APIRouter позволяет вынести маршруты в отдельный файл, чтобы не захламлять main.py.
router = APIRouter()


@router.post(
    "/analyze",
    response_model=CandidateResult,
    status_code=status.HTTP_201_CREATED,
    summary="Анализ кандидата"
)
async def analyze_candidate(
    request: Request,
    file: UploadFile = File(...),
    session: Session = Depends(get_session)
) -> CandidateResult:
    """
    Эндпоинт для обработки резюме/интервью кандидата.

    Принимает файл, сохраняет его, запускает цепочку AI (Mock) -> ML (Mock),
    сохраняет результаты в БД и возвращает сформированный отчет.

    Parameters
    ----------
    file : UploadFile
        Файл, отправленный клиентом через multipart/form-data.
        Может быть аудио (.wav, .mp3) или текст/pdf.
        FastAPI автоматически обрабатывает поток байтов.
    session : Session
        Активная сессия базы данных.
        Внедряется автоматически через Dependency Injection (Depends).

    Returns
    -------
    CandidateResult
        Объект с результатами анализа, включая:
        - Данные кандидата
        - Вектор признаков
        - Оценку удержания (Retention Score)

    Raises
    ------
    HTTPException (500)
        Если произошла внутренняя ошибка сервера (ошибка записи файла, сбой БД, etc).
    """
    try:
        model_ext = request.app.state.extractor
        lock = request.app.state.gpu_lock
        result = await process_candidate(file, session, model_ext, lock)
        return result

    except Exception as e:
        # ======================= NOTE ==========================
        # скорее всего нужен логировщик (logger.error) в релизе.
        # ======================= NOTE ==========================
        print(f"Error processing candidate: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal Server Error: {str(e)}"
        )


@router.get(
    "/history",
    response_model=List[CandidateResult],
    summary="История анализов"
)
def get_history(
    session: Session = Depends(get_session)
) -> List[CandidateResult]:
    """
    Эндпоинт для выгрузки списка всех ранее проанализированных кандидатов.

    Используется для отображения таблицы на дашборде рекрутера.

    Parameters
    ----------
    session : Session
        Активная сессия базы данных.

    Returns
    -------
    List[CandidateResult]
        Список объектов результатов, отсортированный по новизне.

    Raises
    ------
    HTTPException (500)
        При ошибке чтения из базы данных.
    """
    try:
        return get_all_candidates(session)
    except Exception as e:
        print(f"Error fetching history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not fetch history"
        )
