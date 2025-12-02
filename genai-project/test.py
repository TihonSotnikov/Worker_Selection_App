
import json
from app.ai.transcriber import transcriber
from app.ai.extractor import extractor
from app.ai.extractor import SYSTEM_PROMPT_EXTRACT

# transcriber = transcriber("deepdml/faster-whisper-large-v3-turbo-ct2", language="ru")
ext = extractor("Qwen/Qwen3-1.7B")

with open("sample.txt", "r", encoding="utf-8") as f:
    resume_text = f.read()

vector, name, summary = ext(resume_text)
print(name)
print(summary)
print(json.dumps(vector.model_dump(), indent=2, ensure_ascii=False))
