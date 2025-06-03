from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
import time
import os
from src.serving.inference import RecommendationModel, get_recommendation_model
from src.utils.logging_utils import setup_logger
from src.utils.config_loader import load_config
# from prometheus_fastapi_instrumentator import Instrumentator # Optional for Prometheus

logger = setup_logger("FastAPIApp")
config = load_config()
serving_cfg = config['serving']

app = FastAPI(title="Recommendation Service API")

# Optional: Instrument for Prometheus metrics
# Instrumentator().instrument(app).expose(app)

# Dependency Injection for the model
def get_model() -> RecommendationModel:
    return get_recommendation_model() # Uses the lazy-loaded global instance

class RecommendationRequest(BaseModel):
    user_id: int
    # Optional: Provide history for better CB recs
    history: list[int] | None = None
    k: int = serving_cfg['top_k_recommendations']

class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: list[int]
    model_cf: str
    model_cb: str
    stage: str

@app.on_event("startup")
async def startup_event():
    logger.info("API Startup: Initializing model instance...")
    # Trigger loading the model on startup instead of first request
    get_model()
    logger.info("API Startup complete.")

@app.get("/health", status_code=200)
async def health_check():
    # Basic health check
    # Could add checks for model loading status
    model = get_model()
    if model.als_model is not None or model.tfidf_vectorizer is not None:
         return {"status": "ok", "message": "Service is running and models appear loaded."}
    else:
         # Be careful returning 503 if only one model failed but others work
         logger.warning("Health check: Models not fully loaded.")
         return {"status": "degraded", "message": "Service is running but one or more models failed to load."}


@app.post("/recommend", response_model=RecommendationResponse)
async def recommend_items(
    request: RecommendationRequest,
    model: RecommendationModel = Depends(get_model) # Inject model instance
):
    """Generates recommendations for a given user."""
    start_time = time.time()
    logger.debug(f"Received recommendation request for user: {request.user_id}")

    try:
        recs = model.recommend(request.user_id, request.k, request.history)
        duration = time.time() - start_time
        logger.info(f"Recommendation generation took {duration:.4f} seconds.")

        return RecommendationResponse(
            user_id=request.user_id,
            recommendations=recs,
            model_cf=model.cf_model_name,
            model_cb=model.cb_model_name,
            stage=model.model_stage
        )
    except Exception as e:
        logger.error(f"Error processing recommendation for user {request.user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error generating recommendations.")

# To run locally: uvicorn src.serving.api:app --reload --port 8000