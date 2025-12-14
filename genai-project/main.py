import torch
import uvicorn
import gc
import asyncio
import logging
import subprocess
import sys
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.ai.extractor import extractor
from app.api.database import init_db
from app.api.routes import router as api_router
from app.ml.generator import generate_if_needed
from app.ml.predictor import train_if_needed


def start_dashboard():
    """
    Запуск Streamlit дашборда в отдельном процессе
    """

    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))

        dashboard_path = os.path.join(base_dir, "app", "ui", "dashboard.py")

        if not os.path.exists(dashboard_path):
            print(f"ERROR: Файл дашборда не найден по пути: {dashboard_path}")
            return

        cmd = [
            sys.executable, "-m", "streamlit", "run",
            dashboard_path,
            "--server.port=8501",
            "--server.address=0.0.0.0"
        ]

        subprocess.Popen(cmd)
        print(" Dashboard started: http://localhost:8501")

    except Exception as e:
        print(f" Dashboard error: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Менеджер жизненного цикла приложения.

    Код до yield выполняется при старте сервера.
    Код после yield выполняется при сотановки сервера.
    """

    print("Executing startup logic: initializing DB...")
    app.state.logger = logging.getLogger("uvicorn")
    init_db()

    generate_if_needed()
    train_if_needed()

    app.state.gpu_lock = asyncio.Lock()
    async with app.state.gpu_lock:
        app.state.extractor = extractor("Qwen/Qwen3-4B-Instruct-2507", logger=app.state.logger)

    start_dashboard()

    yield

    print("Executing shutdown logic...")
    async with app.state.gpu_lock:
        if app.state.extractor:
            app.state.logger.info("Releasing extractor resources...")
            del app.state.extractor
            gc.collect()
            torch.cuda.empty_cache()


app = FastAPI(
    title="Worker_Selection_App Backend",
    description="MVP API for Blue-Collar Retention AI",
    version="0.1.0",
    lifespan=lifespan
)

# ============== NOTE ================
# В MVP разрешаем всем все (по сути).
# ============== NOTE ================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )
