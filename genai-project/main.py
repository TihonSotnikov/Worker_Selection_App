import torch
import uvicorn
import gc
import asyncio
import logging
import subprocess
import sys
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

TESTING = os.getenv("TESTING", "0") == "1"

from app.api.database import init_db
from app.api.routes import router as api_router
from app.ui_legacy.dashboard_api import router as dashboard_router
from app.ml_legacy.generator import generate_if_needed
from app.ml_legacy.predictor import train_if_needed
from app.core.config import settings

if not TESTING:
    from app.ai.extractor import extractor


BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "app" / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Менеджер жизненного цикла приложения.

    Код до yield выполняется при старте сервера.
    Код после yield выполняется при остановки сервера.
    """

    print("Executing startup logic...")

    if not os.path.exists(settings.UPLOAD_DIR):
        print(f"Creating upload directory: {settings.UPLOAD_DIR}")
        os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

    print("Initializing DB...")
    app.state.logger = logging.getLogger("uvicorn")
    init_db()

    generate_if_needed()
    train_if_needed()

    app.state.gpu_lock = asyncio.Lock()
    if TESTING:
        app.state.extractor = object()
    else:
        async with app.state.gpu_lock:
            app.state.extractor = extractor(
                "Qwen/Qwen3-4B-Instruct-2507", logger=app.state.logger
            )

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
    lifespan=lifespan,
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
app.include_router(dashboard_router, prefix="/api")

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/", include_in_schema=False)
def frontend_index():
    return FileResponse(FRONTEND_DIR / "index.html")

if __name__ == "__main__":
    uvicorn.run("main:app",
        host="127.0.0.1",
        port=8000,
        reload=os.getenv("RELOAD", "0") == "1",
    )