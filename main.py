from fastapi import FastAPI
from router import recommendation
from config.logging import setup_logging

setup_logging()

app = FastAPI()

app.include_router(recommendation.router, prefix="/api/v1", tags=["Recommendation"])
