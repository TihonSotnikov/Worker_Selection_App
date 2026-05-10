import logging
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, status, Request, Response
from sqlmodel import Session
from typing import List

from app.api.database import get_session
from app.services.analyze_service import process_candidate
from app.services.storage_service import get_all_candidates
from app.services import interview_service
from app.core.schemas import CandidateResult

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/analyze",
    response_model=CandidateResult,
    status_code=status.HTTP_201_CREATED,
    summary="Анализ кандидата",
)
async def analyze_candidate(
    request: Request,
    file: UploadFile = File(...),
    session: Session = Depends(get_session),
) -> CandidateResult:
    """
    Эндпоинт для обработки резюме/интервью кандидата.
    """
    try:
        model_ext = request.app.state.extractor
        lock = request.app.state.gpu_lock
        result = await process_candidate(file, session, model_ext, lock)
        return result

    except Exception as e:
        logger.exception(f"Error processing candidate: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal Server Error: {str(e)}",
        )


@router.get(
    "/history", response_model=List[CandidateResult], summary="История анализов"
)
def get_history(session: Session = Depends(get_session)) -> List[CandidateResult]:
    """
    Эндпоинт для выгрузки списка всех ранее проанализированных кандидатов.
    """
    try:
        return get_all_candidates(session)
    except Exception as e:
        logger.exception(f"Error fetching history: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not fetch history",
        )


@router.get("/interview/start", summary="Начать интервью")
async def start_interview():
    """Возвращает первый вопрос интервью."""
    return await interview_service.get_interview_question(0)


@router.post("/interview/submit_answer", summary="Отправить ответ на вопрос")
async def submit_answer(step: int, transcript: str):
    """Принимает текст ответа и возвращает следующий вопрос."""
    return await interview_service.process_interview_answer(step, transcript)


@router.post("/synthesize", summary="Синтез речи (TTS)")
async def synthesize_text(text: str):
    """Преобразует текст в аудио-файл (WAV)."""
    try:
        audio_bytes = interview_service.tts_engine.synthesize(text)
        return Response(content=audio_bytes, media_type="audio/wav")
    except Exception as e:
        logger.exception(f"TTS synthesis failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Synthesis error"
        )
