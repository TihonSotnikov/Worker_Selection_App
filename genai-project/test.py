
import json
import time
# from app.ai.transcriber import transcriber
from app.ai.extractor import extractor
from app.ai.extractor import SYSTEM_PROMPT_SUMMARY
from app.core.schemas import CandidateSummary, CandidateVector
from colorama import init, Fore, Back, Style
init(autoreset=True)

# transcriber = transcriber("deepdml/faster-whisper-large-v3-turbo-ct2", language="ru")
ext = extractor("Qwen/Qwen3-1.7B")

with open("sample.txt", "r", encoding="utf-8") as f:
    resume_text = f.read()

tests = 4
error_count = 0
problems = []
for i in range(16):
    ts = time.time()
    print(f"\nIteration {i+1}:")
    try:
        vector, name, summary = ext(resume_text)
        print(name)
        print(summary)
        print(json.dumps(vector.model_dump(), indent=2, ensure_ascii=False))
    except Exception as e:
        error_count += 1
        problems.append((name, summary, vector, str(e)))
        print("Error during extraction:", str(e))
    print(f"Time taken: {time.time() - ts:.2f} seconds")

print(f'\nCompleted {tests} tests with {error_count} errors, success rate: {1.0 - error_count / tests:.2f}')
for name, summary, vector, error_msg in problems:
    print("\n--- Problematic Case ---")
    print("Name:", name)
    print("Summary:", summary)
    print("Vector:", json.dumps(vector.model_dump(), indent=2, ensure_ascii=False) if vector else "N/A")
    print("Error Message:", error_msg)

# sum = CandidateSummary(
#     full_name="Иван Иванов",
#     raw_summary="Опытный разработчик с 5-летним стажем в области веб-разработки. [...]")

# print('\n\n-------\nSummary\n')
# print(json.dumps(CandidateSummary.model_json_schema(), indent=2, ensure_ascii=False))
# print('\n\n------\nVector\n')
# print(json.dumps(CandidateVector.model_json_schema(), indent=2, ensure_ascii=False))
# print(json.dumps(json.loads(sum.json()), indent=2, ensure_ascii=False))