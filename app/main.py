from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.config import settings
from app.database import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    # this will initialize the db before the server starts
    init_db()
    from app.services import EmbeddingModelLoader

    EmbeddingModelLoader()  # doing this so that we load the model even before accepting traffic
    yield


app = FastAPI(
    title=settings.APP_NAME,
    lifespan=lifespan,
)  # we are importing the app name from the env file itself


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "app_name": settings.APP_NAME,
        "database_configured": bool(settings.DATABASE_URL),
        "qdrant_path_configured": bool(settings.QDRANT_PATH),
    }
