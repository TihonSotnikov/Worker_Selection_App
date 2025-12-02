
import json
from app.ai.transcriber import transcriber
from app.ai.extractor import extractor
from app.ai.extractor import SYSTEM_PROMPT

# transcriber = transcriber("deepdml/faster-whisper-large-v3-turbo-ct2", language="ru")
ext = extractor("Qwen/Qwen3-1.7B")

with open("sample.txt", "r", encoding="utf-8") as f:
    resume_text = f.read()

print(ext(resume_text))
print(json.dumps(ext(resume_text).model_dump(), indent=2, ensure_ascii=False))