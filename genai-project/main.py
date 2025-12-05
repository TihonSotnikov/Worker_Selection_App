import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.ai.extractor import extractor
from app.api.database import init_db
from app.api.routes import router as api_router
from app.core.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Менеджер жизненного цикла приложения.

    Код до yield выполняется при старте сервера.
    Код после yield выполняется при сотановки сервера.
    """

    print("Executing startup logic: initializing DB...")
    init_db()
    app.state.extractor = extractor("Qwen/Qwen3-1.7B")

    yield

    print("Executing shutdown logic...")


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
