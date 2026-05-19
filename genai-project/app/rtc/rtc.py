import asyncio
import fractions
import collections
from logging import getLogger
from typing import Literal, Callable, Awaitable

import numpy as np
import av
import webrtcvad
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import MediaStreamTrack
from silero import silero_tts

from app.ai.models import transcriber, tts_model
from app.core.config import settings


LOGGER = getLogger(settings.LOGGER)

KNOWN_HALLUCINATIONS = [
    "Продолжение следует...",
]


class VoiceActivityDetector:
    def __init__(self, sample_rate=48000, frame_duration_ms=20, agressivity: int = 3, silence_duration_ms = 2000):
        self.vad = webrtcvad.Vad(3) # Агрессивность (3 - самая высокая)
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        
        # Кол-во сэмплов в одном фрейме (например, 960 для 20мс @ 48кГц)
        self.frame_size = int(sample_rate * (frame_duration_ms / 1000))
        
        # Буферы
        self.buffer = collections.deque(maxlen=int(1000 / frame_duration_ms)) # 1 сек буфер
        self.is_speaking = False
        self.speech_frames = []
        self.silence_frames_count = 0
        self.max_silence_frames = int(silence_duration_ms / frame_duration_ms) 

    def process_frame(self, frame_bytes: bytes):
        """
        Возвращает:
        - None: если ничего не произошло
        - ('speech_start', ...): событие начала речи
        - ('speech_end', audio_bytes): готовый кусок для отправки в STT
        """
        # Важно: webrtcvad требует именно такой длины байтов
        if len(frame_bytes) != self.frame_size * 2:
            return None

        is_speech = self.vad.is_speech(frame_bytes, self.sample_rate)

        if is_speech:
            self.silence_frames_count = 0
            if not self.is_speaking:
                self.is_speaking = True
                self.speech_frames = []
                return ('speech_start', None)
            
            self.speech_frames.append(frame_bytes)
        else:
            if self.is_speaking:
                self.silence_frames_count += 1
                self.speech_frames.append(frame_bytes)
                
                if self.silence_frames_count > self.max_silence_frames:
                    self.is_speaking = False
                    audio_data = b"".join(self.speech_frames)
                    self.speech_frames =[]
                    return ('speech_end', audio_data)
        
        return None


class OutgoingAudioTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self):
        super().__init__()
        self.queue = asyncio.Queue(maxsize=50)
        
        # 1. Настройки формата (WebRTC стандарт)
        self.sample_rate = 48000
        self.samples_per_frame = int(self.sample_rate * 0.02) # 960 сэмплов = 20 мс
        self.time_base = fractions.Fraction(1, self.sample_rate)
        
        self.pts = 0 
    
    def _create_silent_frame(self):
        # ВСЕГДА создаем новый фрейм тишины, чтобы не было гонок данных в aiortc
        silence_array = np.zeros((1, self.samples_per_frame), dtype=np.int16)
        frame = av.AudioFrame.from_ndarray(silence_array, format='s16', layout='mono')
        frame.sample_rate = self.sample_rate
        frame.time_base = self.time_base
        return frame

    async def recv(self):
        """
        aiortc вызывает этот метод ~50 раз в секунду, чтобы получить следующий кадр.
        """
        try:
            frame = await asyncio.wait_for(self.queue.get(), timeout=0.1)
        except asyncio.TimeoutError:
            frame = self._create_silent_frame()

        # Синхронизация времени.
        frame.pts = self.pts
        self.pts += frame.samples
        
        return frame


# class SileroTTSProcessor:
#     def __init__(self):
#         self.model = tts_model
#         self.sample_rate = 24000
#         self.resampler = av.AudioResampler(format='s16', layout='mono', rate=48000)

#     def _generate_sync(self, text, speaker):
#         """Блокирующая функция генерации"""
#         audio = self.model.apply_tts(text=text, sample_rate=self.sample_rate, speaker=speaker)
#         return audio.numpy() # numpy array (float32)

#     async def stream_to_queue(
#             self,
#             text,
#             queue: asyncio.Queue,
#             spaeaker: Literal["aidar", "baya", "kseniya", "xenia", "eugene"]
#             ):
#         """
#         Асинхронный метод для запуска генерации и отправки чанков в очередь
#         """
#         audio_data = await asyncio.to_thread(self._generate_sync, text, speaker=spaeaker)
        
#         input_frame = av.AudioFrame.from_ndarray(audio_data.reshape(1, -1), format='flt', layout='mono')
#         input_frame.sample_rate = self.sample_rate
        
#         resampled_frames = self.resampler.resample(input_frame)
        
#         combined_audio = np.concatenate([f.to_ndarray() for f in resampled_frames], axis=1).flatten()
        
#         # Нарезка по 960 семплов (20 мс при 48кГц)
#         samples_per_frame = 960
#         for i in range(0, len(combined_audio), samples_per_frame):
#             chunk = combined_audio[i : i + samples_per_frame]
            
#             # Если кусок короче 960, дополняем нулями (padding)
#             if len(chunk) < samples_per_frame:
#                 padding = np.zeros(samples_per_frame - len(chunk), dtype=np.int16)
#                 chunk = np.concatenate([chunk, padding])
            
#             # Кладем в очередь
#             await queue.put(chunk.astype(np.int16).reshape(1, -1))


async def send_to_stt(audio_bytes):
    # reshape(1, -1) нужен, так как PyAV ожидает 2D массив для аудио (каналы, сэмплы)
    audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16).reshape(1, -1)
    frame = av.AudioFrame.from_ndarray(audio_int16, format='s16', layout='mono')
    frame.sample_rate = 48000
    
    resampler = av.AudioResampler(format='s16', layout='mono', rate=16000)
    resampled_frames = resampler.resample(frame)
    resampled_frames.extend(resampler.resample(None) or [])
    if not resampled_frames:
        return ""
    
    arrays = [f.to_ndarray().flatten() for f in resampled_frames]
    audio_float32 = np.concatenate(arrays).astype(np.float32) / 32768.0
    
    def _transcribe():
        segments, info = transcriber.transcribe(
            audio_float32,
            language="ru",
            beam_size=1,
            best_of=1,
            condition_on_previous_text=False,
            vad_filter=True
        )
        return " ".join([segment.text for segment in segments])

    try:
        # asyncio.to_thread отправляет синхронную задачу в ThreadPoolExecutor
        text = await asyncio.to_thread(_transcribe)
        text = text.strip()
        if not text:
            return ""
        if text in KNOWN_HALLUCINATIONS:
            return ""
        LOGGER.info(f"Распознано: [{text}]")
        return text
    except Exception as e:
        LOGGER.error(f"Ошибка при транскрибации: {e}", exc_info=True)
        return ""

async def process_speech_pipeline(audio_bytes, handler: Callable[[str], Awaitable[None]]):
    """Обертка для безопасного вызова STT и дальнейшей передачи в LLM"""
    text = await send_to_stt(audio_bytes)
    text = text.strip() if text else ""
    
    if len(text) > 3:
        print(f"Распознано: {text}")
        await handler(text)

async def consume_incoming_audio(
        track: MediaStreamTrack,
        outgoing_queue: asyncio.Queue,
        fragment_handler: Callable[[str], Awaitable[None]],
        lock: asyncio.Lock | None = None
        ):
    """
    Фоновая задача, которая непрерывно читает микрофон пользователя.
    """
    LOGGER.info("Начато чтение входящего аудио потока...")
    
    vad_processor = VoiceActivityDetector()
    resampler = av.AudioResampler(
        format='s16', 
        layout='mono', 
        rate=48000
    )
    stt_queue = asyncio.Queue()

    async def stt_worker():
        while True:
            audio_data = await stt_queue.get()
            await process_speech_pipeline(audio_data, fragment_handler)
            stt_queue.task_done()
    worker_task = asyncio.create_task(stt_worker())

    try:
        while True:
            frame: av.AudioFrame = await track.recv() # type: ignore
            
            # --- PROCESSING LOGIC ---
            resampled = resampler.resample(frame)
            if not resampled:
                continue

            frame_bytes = resampled[0].to_ndarray().tobytes()
            event = vad_processor.process_frame(frame_bytes)
            # out_frame = resampled[0]

            if event:
                event_type, data = event
                if event_type == 'speech_start':
                    ...
                elif event_type == 'speech_end':
                    # FRAGMENT PROCESSING #
                    stt_queue.put_nowait(data)
                    # /FRAGMENT PROCESSING #
            
            # --- END PROCESSING LOGIC ---
            
            # if outgoing_queue.full():
            #     try:
            #         outgoing_queue.get_nowait()
            #     except asyncio.QueueEmpty:
            #         pass
            # await outgoing_queue.put(out_frame)
            
    except Exception as e:
        LOGGER.info(f"Входящий аудио поток завершен: {e}")
    finally:
        worker_task.cancel()

async def stream_text_to_speech(
        text: str,
        queue: asyncio.Queue,
        speaker: Literal["aidar", "baya", "kseniya", "xenia", "eugene"]
        ):
    
    resampler = av.AudioResampler(
        format='s16', 
        layout='mono', 
        rate=48000
    )
    raw_audio_chunks = tts_model.generate_stream(text)
    resampler = av.AudioResampler(format='s16', layout='mono', rate=48000)
    
