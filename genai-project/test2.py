import json
from pprint import pprint
from app.ai.extractor import extractor
from app.ai.extractor import SYSTEM_PROMPT_EXTRACT
from app.core.schemas import CandidateSummary, CandidateVector
from colorama import init, Fore, Back, Style
from time import time
init(autoreset=True)

ext = extractor("Qwen/Qwen3-1.7B")
candidatesummary_schema = CandidateSummary.model_json_schema()
candidatesummary_json_schema = json.dumps(CandidateSummary.model_json_schema(), indent=2, ensure_ascii=False)

print(len(ext._pipeline.tokenizer.tokenize(SYSTEM_PROMPT_EXTRACT)))

print('\n\n-------\nSummary\n')
with open("sample.txt", "r", encoding="utf-8") as f:
    resume_text = f.read()

t = time()
vec, name, summary = ext(resume_text)
print(f"Extraction took {time() - t:.2f} seconds.\n")
print(Fore.GREEN + name)
print(Fore.LIGHTBLUE_EX + summary)
print(Fore.LIGHTMAGENTA_EX + vec.model_dump_json(indent=2,ensure_ascii=False) + Fore.RESET)
# print(candidatesummary_json_schema)
# print("CandidateSummary JSON Schema Tokens count:", len(ext._pipeline.tokenizer.tokenize(candidatesummary_json_schema)))
# print('\n\n------\nVector\n')
# print(json.dumps(CandidateVector.model_json_schema(), indent=2, ensure_ascii=False))
# print(json.dumps(json.loads(sum.json()), indent=2, ensure_ascii=False))