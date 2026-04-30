# 🚀 MVP ПЛАН (2 недели, 1-15 мая)

## 📋 ПРОБЛЕМЫ (текущее состояние)

- **Архитектура туда-сюда** — нужно чистить (слои, импорты, ошибки)
- **TTS нет** — для Андрея звонит → текст → LLM → ответ → озвучить
- **UI на Streamlit** — заменить на HTML+CSS+JS (для Ильи)
- **ML слабая** — только 6 признаков, ROC-AUC ~0.75

---

## 🎯 ЗАДАЧИ ПО ЛЮДЯМ

### ТИХОН (Архитектура + Backend)
**Неделя 1 (1-5 мая):**
1. Создать `app/services/` layer
   - `analyze_service.py` — главный оркестратор (берет файл → вызывает AI → ML → storage)
   - `ai_service.py` — обертка над STT + LLM (принимает файл, возвращает структурированные данные)
   - `ml_service.py` — обертка над CatBoost (принимает вектор, возвращает score + risks)
   - `storage_service.py` — работа с БД
2. Рефакторить `app/api/routes.py` — использовать services вместо прямых импортов
3. Убрать Streamlit из main.py (запускаться отдельно: `streamlit run app/ui_legacy/dashboard.py`)
4. Добавить логирование + error handling (если AI не доступна → graceful fallback)
5. Убедиться что тесты проходят

**Неделя 2 (12-15 мая):**
- [ ] Проверить, что все endpoints работают (`/analyze`, `/history`, `/synthesize`, `/interview/*`)
- [ ] Docker image оптимизировать (минимальный размер)
- [ ] README обновить (как запустить локально и в Docker)

**Code пример:**
```python
# routes.py (after)
from app.services.analyze_service import AnalyzeService

@router.post("/analyze")
async def analyze(file: UploadFile):
    service = AnalyzeService()
    result = await service.process(file)  # All logic here
    return result
```

---

### АНДРЕЙ (Audio-to-Audio)
**Неделя 1-2 (2-9 мая):**
1. Выбрать TTS (Silero TTS — легкая, offline, русский язык ✓)
2. Создать `app/ai/tts.py`
   ```python
   class TextToSpeechEngine:
       def synthesize(self, text: str) -> bytes:  # returns WAV audio
   ```
3. Добавить эндпоинты в `app/api/routes.py`:
   - `POST /api/synthesize` — text → audio
   - `POST /api/interview/start` — start mock interview, return first question (озвученный)
   - `POST /api/interview/submit_answer` — submit answer (audio), get next question или result
4. Создать `app/ai/interviewer.py` — симуляция интервьюера
   - Хранит 5 вопросов, собирает ответы, в конце передает LLM для обработки
5. Написать unit тесты (TTS работает, STT работает, full cycle работает)
6. Создать 3 demo audio примера (или озвучить синтезом)

**Output**: Audio-to-Audio pipeline работает end-to-end

---

### ИЛЬЯ (ML + Frontend)
**Неделя 1 (1-5 мая) — ML:**
1. Расширить датасет (добавить признаки):
   - Текущие: `skills_verified_count`, `years_experience`, `commute_time_minutes`, `shift_preference`, `salary_expectation`, `has_certifications`
   - Новые: `education_level`, `previous_turnovers`, `family_status`, `housing_type`, `has_transport`
2. Обновить `SyntheticDataGenerator` с новыми правилами (if previous_turnovers > 3 → high risk, if has_kids + night_shift → risky, etc.)
3. Переучить CatBoost на новых данных (ROC-AUC > 0.80)
4. Улучшить `explain_prediction()` — возвращать не только `risk_factors`, но и `positive_factors`

**Неделя 2 (6-14 мая) — Frontend:**
1. HTML+CSS+JS интерфейс (vanilla JS, no frameworks)
   - Landing page (title + button "Start")
   - Upload page (drag-drop file or mic input)
   - Processing page (spinner + status)
   - Results page (big retention score circle, red/yellow/green, risk factors, positive factors)
   - History page (table of all candidates + demo mode buttons)
2. Интеграция с API (fetch POST /api/analyze, GET /api/history)
3. Responsive design (работает на мобильных)
4. 3 demo candidates встроены (кнопки "Load Green/Yellow/Red")

**Output**: Beautiful web UI, no Streamlit

---

## 📅 TIMELINE

```
1-5 мая      Тихон архитектура + Андрей TTS + Илья ML
6-9 мая      Андрей audio finish + Илья фронтенд
10-12 мая    Интеграция + тестирование
13-15 мая    Полировка + демо + готово к защите
```

---

## ✅ ГОТОВО КОГДА:

- ✅ Backend: services layer ready, все endpoints работают, тесты проходят
- ✅ Audio: TTS + STT + LLM → озвученный interview работает
- ✅ Frontend: upload → analyze → красивый результат (retention score + risks)
- ✅ ML: ROC-AUC > 0.80, плюсы и минусы объясняются
- ✅ Demo: 3 сценария (Green/Yellow/Red) работают в UI
- ✅ DevOps: Docker запускается, README есть
- [ ] No critical bugs

---

## 🛑 CRITICAL PATH

**Тихон → все остальное** — если архитектура отстанет на 2 дня, весь проект отстанет.

Приоритет Неделя 1 Тихона:
1. services layer (день 1-2)
2. UI отделение (день 2)
3. error handling + логи (день 3)
4. tests (день 4)

---

## 💬 COMMUNICATION & TRACKING

- **Daily standup**: 10:00, 15 минут
- **Format**: "Вчера X. Сегодня Y. Блокеры: Z?"
- **Git commits**: каждый день в master (или branch + PR)
- **Weekly check (Пятница)**: Все ли идет по плану? Нужна ли корректировка?

---

## 📦 DELIVERABLES (15 мая)

- ✅ Code: `genai-project/` весь готов
- ✅ Docs: `README.md`, `requirements.txt`
- ✅ Deployment: `Dockerfile`, `docker-compose.yml` работают
- ✅ UI: Beautiful frontend, no Streamlit
- ✅ Demo: 3 сценария готовы к демонстрации на защите

---

**Начинаем завтра (1 мая) в 10:00. Let's go! 🚀**
