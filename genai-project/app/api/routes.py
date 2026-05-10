import logging
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, status, Request
from sqlmodel import Session
from typing import List

from app.api.database import get_session
from app.services.analyze_service import process_candidate
from app.services.storage_service import get_all_candidates
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
