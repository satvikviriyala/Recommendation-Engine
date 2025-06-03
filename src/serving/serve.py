import os
import yaml
import mlflow
import logging
from pathlib import Path
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import scipy.sparse
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALSModel
import joblib
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Recommendation Engine API")

# Load configuration
def load_config():
    """Load configuration from yaml file."""
    config_path = Path("configs/config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)

# Initialize Spark
def init_spark():
    """Initialize Spark session."""
    return (SparkSession.builder
            .appName("RecSysServing")
            .config("spark.executor.memory", "4g")
            .config("spark.driver.memory", "2g")
            .getOrCreate())

# Load models
def load_models():
    """Load both collaborative filtering and content-based models."""
    config = load_config()
    
    # Load ALS model
    cf_model_path = os.path.expandvars(config['als_model']['output_path'])
    cf_model = ALSModel.load(cf_model_path)
    
    # Load TF-IDF model
    vectorizer_path = os.path.expandvars(config['tfidf_model']['output_vectorizer_path'])
    matrix_path = os.path.expandvars(config['tfidf_model']['output_matrix_path'])
    
    vectorizer = joblib.load(vectorizer_path)
    matrix_data = np.load(matrix_path)
    tfidf_matrix = scipy.sparse.csr_matrix(
        (matrix_data['data'], matrix_data['indices'], matrix_data['indptr']),
        shape=matrix_data['shape']
    )
    
    return cf_model, vectorizer, tfidf_matrix

# Request/Response models
class RecommendationRequest(BaseModel):
    user_id: int
    num_recommendations: int = 10
    use_content_based: bool = False

class RecommendationResponse(BaseModel):
    recommendations: List[Dict[str, Any]]
    model_used: str

# Global variables
config = load_config()
spark = init_spark()
cf_model, vectorizer, tfidf_matrix = load_models()

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.post("/recommend", response_model=RecommendationResponse)
def get_recommendations(request: RecommendationRequest):
    """Get recommendations for a user."""
    try:
        if request.use_content_based:
            # Content-based recommendations
            user_profile = get_user_profile(request.user_id)
            scores = tfidf_matrix.dot(user_profile.T)
            top_items = np.argsort(scores.toarray().flatten())[-request.num_recommendations:][::-1]
            
            recommendations = [
                {
                    "item_id": int(item_id),
                    "score": float(scores[0, item_id])
                }
                for item_id in top_items
            ]
            model_used = "content_based"
            
        else:
            # Collaborative filtering recommendations
            user_df = spark.createDataFrame([(request.user_id,)], ["user_id"])
            predictions = cf_model.recommendForUserSubset(user_df, request.num_recommendations)
            
            recommendations = [
                {
                    "item_id": int(row.item_id),
                    "score": float(row.rating)
                }
                for row in predictions.collect()[0].recommendations
            ]
            model_used = "collaborative_filtering"
        
        return RecommendationResponse(
            recommendations=recommendations,
            model_used=model_used
        )
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def get_user_profile(user_id: int):
    """Get user profile for content-based recommendations."""
    # This is a placeholder - implement actual user profile generation
    # based on user's interaction history
    return np.ones((1, tfidf_matrix.shape[1]))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=config['serving']['host'],
        port=config['serving']['port']
    ) 