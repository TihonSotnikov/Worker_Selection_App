import asyncio
import fractions
from logging import getLogger

import numpy as np
import av
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import MediaStreamTrack

from app.ai.transcriber import transcriber
from app.core.config import settings


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


async def consume_incoming_audio(track: MediaStreamTrack, outgoing_queue: asyncio.Queue):
    """
    Фоновая задача, которая непрерывно читает микрофон пользователя.
    """
    logger = getLogger(settings.LOGGER)
    logger.info("Начато чтение входящего аудио потока...")
    
    resampler = av.AudioResampler(
        format='s16', 
        layout='mono', 
        rate=48000
    )

    try:
        while True:
            frame: av.AudioFrame = await track.recv() # type: ignore
            
            # --- PROCESSING LOGIC ---
            resampled_frames = resampler.resample(frame)
            if not resampled_frames:
                continue
            out_frame = resampled_frames[0]
            # audio_data = frame.to_ndarray()
            # audio_data = (audio_data.astype(np.int32) * 1.3)
            # audio_data = np.clip(audio_data, -32768, 32767).astype(np.int16)

            # out_frame = av.AudioFrame.from_ndarray(audio_data, format='s16', layout='mono') 
            # out_frame.sample_rate = frame.sample_rate
            # out_frame.time_base = frame.time_base
            # --- END PROCESSING LOGIC ---
            
            if outgoing_queue.full():
                try:
                    outgoing_queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            await outgoing_queue.put(out_frame)
            
    except Exception as e:
        logger.info(f"Входящий аудио поток завершен: {e}")
