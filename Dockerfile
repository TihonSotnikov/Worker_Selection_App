# Используем стабильный slim образ
FROM python:3.11-slim-bookworm

# Установка системных зависимостей (ffmpeg обязателен для Whisper)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Установка uv из официального бинарника
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uv/bin/uv

# Настройка рабочего окружения
WORKDIR /app

# Копируем файлы зависимостей
COPY pyproject.toml uv.lock ./

# Устанавливаем зависимости в системный Python (для контейнера это ок)
# Используем --frozen для гарантии соответствия uv.lock
RUN /uv/bin/uv sync --frozen --no-dev --no-install-project

# Копируем всё содержимое проекта
COPY . .

# Прокидываем PYTHONPATH на папку с кодом, чтобы импорты внутри genai-project работали
ENV PYTHONPATH=/app/genai-project

# Открываем порт
EXPOSE 8000

# Запуск через uvicorn
# Мы указываем путь к приложению относительно PYTHONPATH
CMD ["/app/.venv/bin/python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
