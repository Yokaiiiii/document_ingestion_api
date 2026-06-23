from fastapi import FastAPI
from app.config import settings

app = FastAPI(
    title=settings.APP_NAME
)  # we are importing the app name from the env file itself


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "app_name": settings.APP_NAME,
    }
