import asyncio
from typing import Callable, Awaitable

from transformers import PreTrainedTokenizerBase, PreTrainedModel, BatchEncoding, GenerationConfig

from app.ai.models import gen_config


SYSTEM_PROMPT = """
Ты - рекрутёр, проводящий интервью с кандидатом.
Ты должен задавать вопросы кандидату, касающиеся опыта, навыков и предпочтений.
Задавай вопросы по одному, не выдавая сразу весь список. Будь краток.
После ответа (если имеет смысл) уточняй, всё ли это, или есть что добавить.
Проговори следующие темы:
- Опыт работы (компании, должности, обязанности)
- Навыки (технические, мягкие)
- Предпочтения по работе (график, удалёнка, зарплата)
Не используй форматирование Markdown, вместо этого пиши в обычном литературном формате.
Если пора закончить интервью, добавь в свой ответ `<stop>`.
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
    
    def stream_response(self, prompt):
        ...
    
    async def stream_from_queue(self, handler: Callable[[str], Awaitable[None]]):
        while True:
            text = await self.input_queue.get()
            if text == None:
                break
            response = await asyncio.to_thread(self.__call__, text)
            await handler(response)
