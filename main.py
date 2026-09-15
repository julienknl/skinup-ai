from fastapi import FastAPI
from router import recommendation

app = FastAPI()

app.include_router(recommendation.router, prefix="/api/v1", tags=["Recommendation"])
