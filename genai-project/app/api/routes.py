import json
import asyncio
from logging import Logger, getLogger

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, WebSocketDisconnect, status, Request, WebSocket
from sqlmodel import Session
from typing import List
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.sdp import candidate_from_sdp
from aiortc.mediastreams import MediaStreamTrack

from app.api.database import get_session
from app.api.services import process_candidate, get_all_candidates
from app.core.schemas import CandidateResult
from app.core.config import settings
from app.rtc.rtc import OutgoingAudioTrack, consume_incoming_audio

# APIRouter позволяет вынести маршруты в отдельный файл, чтобы не захламлять main.py.
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
    logger = getLogger(settings.LOGGER)

    try:
        model_ext = request.app.state.extractor
        lock = request.app.state.gpu_lock
        result = await process_candidate(file, session, model_ext, lock)
        return result

    except Exception as e:
        # ======================= NOTE ==========================
        # скорее всего нужен логировщик (logger.error) в релизе.
        # ======================= NOTE ==========================
        logger.error(f"Error processing candidate: {e}")
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
    logger = getLogger(settings.LOGGER)
    try:
        return get_all_candidates(session)
    except Exception as e:
        logger.error(f"Error fetching history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not fetch history",
        )

@router.websocket("/ws/webrtc")
async def webrtc_endpoint(websocket: WebSocket):
    logger = getLogger(settings.LOGGER)

    await websocket.accept()
    pc = RTCPeerConnection()

    outgoing_track = OutgoingAudioTrack()
    pc.addTrack(outgoing_track)
    background_tasks = set()
    
    @pc.on("track")
    def on_track(track: MediaStreamTrack):
        if track.kind == "audio":
            task = asyncio.create_task(consume_incoming_audio(track, outgoing_track.queue))
            background_tasks.add(task)
            task.add_done_callback(background_tasks.discard)
               
    try:
        data = await websocket.receive_text()
        offer_dict = json.loads(data)
        offer = RTCSessionDescription(sdp=offer_dict["sdp"], type=offer_dict["type"])
        await pc.setRemoteDescription(offer)

        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        await websocket.send_text(json.dumps({
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type
        }))

        # Основной цикл
        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                
                if message.get("type") == "ice-candidate":
                    candidate_info = message.get("candidate")
                    
                    # Проверяем, что кандидат существует и не является пустой строкой
                    # (браузер шлет пустую строку, чтобы сказать "кандидаты закончились")
                    if candidate_info and candidate_info.get("candidate"):
                        candidate = candidate_from_sdp(candidate_info["candidate"])
                        candidate.sdpMid = candidate_info.get("sdpMid")
                        candidate.sdpMLineIndex = candidate_info.get("sdpMLineIndex")
                        await pc.addIceCandidate(candidate)
                        
            except json.JSONDecodeError:
                pass # Игнорируем невалидный JSON
            except Exception as parse_err:
                logger.error(f"Ошибка при разборе сообщения от клиента: {parse_err}")
                
    except WebSocketDisconnect:
        logger.info("WebSocket: Клиент отключился (WebSocketDisconnect)")
    except Exception as e:
        logger.error(f"WebSocket: Произошла непредвиденная ошибка: {e}")
    finally:
        logger.info("Закрытие RTCPeerConnection и остановка задач...")
        await pc.close()
        
        for task in background_tasks:
            task.cancel()
