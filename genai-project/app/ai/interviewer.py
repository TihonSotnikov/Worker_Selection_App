import asyncio
import threading
import os
from typing import Callable, Awaitable

import torch
from transformers import PreTrainedTokenizerBase, PreTrainedModel, BatchEncoding, GenerationConfig, TextIteratorStreamer

from app.ai.models import gen_config


DIRNAME = os.path.dirname(__file__)
SYSTEM_PROMPT = """
Ты - ИИ-рекрутёр, проводящий интервью с кандидатом.
Ты должен задавать вопросы кандидату, касающиеся опыта, навыков и предпочтений.
Задавай вопросы по одному, не выдавая сразу весь список. Будь краток.
Никак не комментируй упоминаемые кандидатом компании в целом.
После ответа (если имеет смысл) уточняй, всё ли это, или есть что добавить.
Проговори следующие темы:
- Опыт работы (компании, должности, обязанности)
- Навыки (технические, мягкие)
- Образование, подтверждающие сертификаты
- Текущее место проживания (чем точнее тем лучше)
- На каких должностях согласен работать
- Предпочтения по работе (график, удалёнка, зарплата)
Не используй форматирование Markdown, вместо этого пиши в обычном литературном формате.
Если пора закончить интервью, добавь в свой ответ `<stop>`.
"""

DIALOG_INIT_PROMPT = """
[SYSTEM] Ты - инициатор этого интервью. Представься, попроси представиться собеседника и начни интервью.
"""

GEN_CFG = GenerationConfig(
    max_new_tokens = 256,
    do_sample = False,
    temperature = 1.0,
    repetition_penalty = 1.0
)


class Interviewer:
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, *args, **kwargs):
        self._model = model
        self._tokenizer = tokenizer
        self.system_prompt = SYSTEM_PROMPT
        self.input_queue = asyncio.Queue()
        self.conversation_history = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        self.log = []
    
    def __call__(self, prompt, *args, **kwds):
        self.conversation_history.append({"role": "user", "content": prompt})
        
        inputs: BatchEncoding = self._tokenizer.apply_chat_template(
            self.conversation_history,
            return_tensors="pt",
            enable_thinking=False,
            add_generation_prompt=True
        ) # type: ignore
        inputs.to(self._model.device)
        
        output_tokens = self._model.generate(**inputs, generation_config=GEN_CFG) # type: ignore
        response_tokens = output_tokens[0][inputs["input_ids"].shape[1]:]
        response_text: str = self._tokenizer.decode(response_tokens, skip_special_tokens=True) # type: ignore

        self.conversation_history.append({"role": "assistant", "content": response_text})
        return response_text
    
    async def stream_response(self, prompt: str):
        self.conversation_history.append({"role": "user", "content": prompt})
        
        inputs: BatchEncoding = self._tokenizer.apply_chat_template(
            self.conversation_history,
            return_tensors="pt",
            enable_thinking=False,
            add_generation_prompt=True
        ) # type: ignore
        inputs = inputs.to(self._model.device)
        
        streamer = TextIteratorStreamer(self._tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            generation_config=GEN_CFG
        )
        
        thread = threading.Thread(target=self._model.generate, kwargs=generation_kwargs) # type: ignore
        thread.start()
        
        full_text = ""
        iterator = iter(streamer)
        while True:
            try:
                token = await asyncio.to_thread(next, iterator, None)
                if token is None:
                    break
                full_text += token
                yield token
            except StopIteration:
                break
        self.conversation_history.append({"role": "assistant", "content": full_text})
    
    async def stream_from_queue(
            self,
            handler: Callable[[str, bool], Awaitable[None]],
            turn_lock: asyncio.Lock | None = None):
        while True:
            text = await self.input_queue.get()
            if text is None:
                break
            
            if turn_lock is not None:
                async with turn_lock:
                    async for chunk in self.stream_response(text):
                        await handler(chunk, False)
                    await handler("", True)
            else:
                async for chunk in self.stream_response(text):
                    await handler(chunk, False)
                await handler("", True)
