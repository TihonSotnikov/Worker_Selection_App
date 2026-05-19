import asyncio
import uuid
import json
import logging
from pathlib import Path
from typing import Tuple
from logging import getLogger

from fastapi import UploadFile, HTTPException, WebSocket, WebSocketDisconnect, status
from fastapi.concurrency import run_in_threadpool
from sqlmodel import Session, select
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer
from aiortc.sdp import candidate_from_sdp
from aiortc.mediastreams import MediaStreamTrack
from transformers import TextIteratorStreamer
from app.rtc.rtc import OutgoingAudioTrack, consume_incoming_audio
from app.ai.models import llm_model, llm_tokenizer, gen_config, tts_model
from app.ai.interviewer import Interviewer

from app.core.config import settings
from app.core.schemas import CandidateVector, CandidateResult
from app.core.enums import ShiftPreference
from app.api.models_db import CandidateTable
from app.ai.extractor import Extractor
from app.ai.transcriber import Transcriber
from app.core.types import AdvancedLock
from app.ai.models import gpu_lock
from app.ai.interviewer import DIALOG_INIT_PROMPT


logger = getLogger('uvicorn')

# logging.basicConfig(level=logging.DEBUG)
# logging.getLogger('aioice').setLevel(logging.DEBUG)
# logging.getLogger('aiortc').setLevel(logging.DEBUG)


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

    file_extension = Path(upload_file.filename).suffix # type: ignore
    unique_filename = f"{uuid.uuid4()}{file_extension}"

    file_path = Path(settings.UPLOAD_DIR) / unique_filename
    file_path.parent.mkdir(parents=True, exist_ok=True)

    content = await upload_file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    return file_path


async def ai_extract(
    file_path: Path, ext: Extractor
        ) -> Tuple[str, str, CandidateVector]:
    """AI экстракция данных из резюме."""

    extension = file_path.suffix.lower()

    if extension in [".wav", ".mp3"]:
        try:
            stt_model = Transcriber("medium")
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


async def ml_predict(vector: CandidateVector) -> Tuple[float, list[str]]:
    """Вызов ML-модуля для предсказания удержания кандидата."""

    try:
        from app.ml_legacy.predictor import RetentionPredictor

        predictor = RetentionPredictor()
        predictor.load_model()

        if predictor.model is None:
            predictor.train_model()
            predictor.save_model()

        features = {
            "skills_verified_count": vector.skills_verified_count,
            "years_experience": vector.years_experience,
            "commute_time_minutes": vector.commute_time_minutes,
            "shift_preference": vector.shift_preference.value,
            "salary_expectation": vector.salary_expectation,
            "has_certifications": vector.has_certifications,
        }

        prediction = predictor.predict_retention(features)
        risk_factors = predictor.explain_prediction(features)

        return float(prediction["retention_probability"]), risk_factors

    except Exception as e:
        print(f"ML модуль недоступен: {e}")

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


async def process_candidate(
    upload_file: UploadFile,
    session: Session,
    model_ext: Extractor,
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

    full_name, raw_summary, vector = await ai_extract(file_path, model_ext)

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
        vec_has_certifications=vector.has_certifications,
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
        risk_factors=json.loads(db_candidate.risk_factors),
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
    list[CandidateResult]
        Список всех кандидатов, отсортированных по дате (новые первые).
    """

    candidates = session.exec(
        select(CandidateTable).order_by(CandidateTable.created_at.desc()) # type: ignore
    ).all()

    results = []
    for db_candidate in candidates:
        vector = CandidateVector(
            skills_verified_count=db_candidate.vec_skills_count,
            years_experience=db_candidate.vec_years_experience,
            commute_time_minutes=db_candidate.vec_commute_minutes,
            shift_preference=ShiftPreference(db_candidate.vec_shift_preference),
            salary_expectation=db_candidate.vec_salary_expectation,
            has_certifications=db_candidate.vec_has_certifications,
        )

        result = CandidateResult(
            id=str(db_candidate.id),
            full_name=db_candidate.full_name,
            raw_summary=db_candidate.raw_summary,
            vector=vector,
            retention_score=db_candidate.retention_score,
            risk_factors=json.loads(db_candidate.risk_factors),
        )
        results.append(result)

    return results


async def handle_webrtc_session(websocket: WebSocket):

    if gpu_lock.locked():
        await websocket.accept()
        await websocket.close(status.WS_1013_TRY_AGAIN_LATER, "Кто-то уже проходит интервью.")
        return
    
    async with gpu_lock:
        await websocket.accept()

        rtc_config = RTCConfiguration(
            iceServers=[
                RTCIceServer(urls=[
                    "stun:stun.cloudflare.com:3478",
                    "stun:stun.sipnet.ru:3478",
                    "stun:stun.nextcloud.com:443",
                ])
            ]
        )
        pc = RTCPeerConnection(configuration=rtc_config)
        interviewer = Interviewer(llm_model, llm_tokenizer)
        turn_lock = AdvancedLock()
        
        # Блокировка для предотвращения RuntimeError при одновременной записи в WebSocket
        ws_lock = asyncio.Lock()

        outgoing_track = OutgoingAudioTrack()
        pc.addTrack(outgoing_track)
        background_tasks: set[asyncio.Task] = set()
        
        send_buffer = ""            # Буфер для выявления <stop>
        full_current_response = ""  # Вся текущая реплика
        is_interview_over = False   # Флаг конца диалога

        async def safe_ws_send(message: dict):
            """Безопасная отправка в сокет, защищающая от race conditions."""
            async with ws_lock:
                try:
                    await websocket.send_text(json.dumps(message))
                except Exception as e:
                    logger.error(f"Ошибка отправки в WebSocket: {e}")

        async def put_to_interviewer_queue(text: str):
            if is_interview_over:
                return
            
            if turn_lock.locked():
                logger.info(f"Бот говорит. Игнорируем реплику пользователя: {text}")
                return
            
            while not interviewer.input_queue.empty():
                try:
                    interviewer.input_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

            await safe_ws_send({"type": "interview-chunk", "content": text})
            interviewer.input_queue.put_nowait(text)
        
        async def handle_interviewer_response(chunk: str, is_final: bool):
            nonlocal send_buffer, full_current_response, is_interview_over
            
            if is_interview_over:
                if is_final:
                    if full_current_response.strip():
                        await safe_ws_send({
                            "type": "interview-finalize",
                            "content": f"Интервьюер: — {full_current_response.strip()}"
                        })
                    await safe_ws_send({"type": "interview-end"})
                return

            send_buffer += chunk
            full_current_response += chunk
            
            if "<stop>" in send_buffer:
                is_interview_over = True
                
                send_buffer = send_buffer.split("<stop>")[0]
                full_current_response = full_current_response.split("<stop>")[0]
                
                if send_buffer:
                    await safe_ws_send({"type": "interview-chunk", "content": send_buffer})
                    send_buffer = ""
            else:
                hold_len = 0
                for i in range(1, 6):
                    if send_buffer.endswith("<stop>"[:i]):
                        hold_len = i
                        break
                
                if hold_len > 0:
                    to_send = send_buffer[:-hold_len]
                    send_buffer = send_buffer[-hold_len:]
                else:
                    to_send = send_buffer
                    send_buffer = ""
                    
                if to_send:
                    await safe_ws_send({"type": "interview-chunk", "content": to_send})
            
            if is_final:
                if send_buffer:
                    await safe_ws_send({"type": "interview-chunk", "content": send_buffer})
                    send_buffer = ""

                if full_current_response.strip():
                    await safe_ws_send({
                        "type": "interview-finalize",
                        "content": f"Интервьюер: — {full_current_response.strip()}"
                    })
                
                full_current_response = ""
                
                if is_interview_over:
                    await safe_ws_send({"type": "interview-end"})
                    try:
                        await websocket.close(1000, "Interview Finished")
                    except Exception:
                        pass
        
        @turn_lock.on_acquire()
        async def on_response_start():
            await safe_ws_send({"type": "interview-model"})

        @turn_lock.on_release()
        async def on_response_end():
            await safe_ws_send({"type": "interview-user"})

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            if pc.connectionState == "connected":
                logger.info("Связь установлена.")
                interviewer.input_queue.put_nowait(DIALOG_INIT_PROMPT)
                task = asyncio.create_task(interviewer.stream_from_queue(handle_interviewer_response, turn_lock))
                background_tasks.add(task)
                task.add_done_callback(background_tasks.discard)
            elif pc.connectionState in ["failed", "closed"]:
                logger.info("Соединение разорвано или не удалось.")
                try:
                    await websocket.close(1000, "WebRTC connection failed")
                except Exception:
                    return
        
        @pc.on("track")
        def on_track(track: MediaStreamTrack):
            if track.kind == "audio":
                # ВАЖНО: Мы передаем None вместо turn_lock.
                # Фильтрация перебиваний вынесена в put_to_interviewer_queue.
                # Если передать lock в вашу текущую версию consume_incoming_audio,
                # случится UnboundLocalError переменной `frame`.
                task = asyncio.create_task(consume_incoming_audio(
                    track,
                    outgoing_track.queue,
                    put_to_interviewer_queue
                ))
                background_tasks.add(task)
                task.add_done_callback(background_tasks.discard)
                
        try:
            data = await websocket.receive_text()
            offer_dict = json.loads(data)
            offer = RTCSessionDescription(sdp=offer_dict["sdp"], type=offer_dict["type"])
            await pc.setRemoteDescription(offer)

            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)
            for _ in range(20):
                if pc.iceGatheringState == "complete":
                    break
                await asyncio.sleep(0.1)
            
            print("Готовый SDP Backend'а:")
            for line in pc.localDescription.sdp.splitlines():
                if "a=candidate" in line:
                    print(line)

            await safe_ws_send({
                "sdp": pc.localDescription.sdp,
                "type": pc.localDescription.type
            })

            # Основной цикл сигналлинга WebRTC
            while True:
                data = await websocket.receive_text()
                try:
                    message = json.loads(data)
                    if message.get("type") == "ice-candidate":
                        candidate_info = message.get("candidate")
                        if candidate_info and candidate_info.get("candidate"):
                            candidate = candidate_from_sdp(candidate_info["candidate"])
                            candidate.sdpMid = candidate_info.get("sdpMid")
                            candidate.sdpMLineIndex = candidate_info.get("sdpMLineIndex")
                            await pc.addIceCandidate(candidate)
                            
                except json.JSONDecodeError:
                    pass
                except Exception as parse_err:
                    logger.error(f"Ошибка при разборе сообщения от клиента: {parse_err}")
                    
        except WebSocketDisconnect:
            logger.info("WebSocket: Клиент отключился (WebSocketDisconnect)")
        except Exception as e:
            logger.error(f"WebSocket: Произошла непредвиденная ошибка: {e}")
        finally:
            logger.info("Закрытие RTCPeerConnection и остановка задач...")
            await pc.close()
            # Отменяем фоновые таски (генерацию LLM и прослушивание аудио)
            for task in background_tasks:
                task.cancel()
