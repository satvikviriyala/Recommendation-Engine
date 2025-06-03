import mlflow
import mlflow.pyfunc
import mlflow.spark
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import scipy.sparse as sp
import os
from src.utils.config_loader import load_config
from src.utils.logging_utils import setup_logger

logger = setup_logger("InferenceService")
config = load_config()

class RecommendationModel:
    """Handles loading models and generating recommendations."""

    def __init__(self, cf_model_name, cb_model_name, model_stage="Production"):
        self.cf_model_name = cf_model_name
        self.cb_model_name = cb_model_name
        self.model_stage = model_stage
        self.als_model = None
        self.tfidf_vectorizer = None
        self.item_tfidf_matrix = None
        self.user_map = None
        self.item_map = None
        self.item_id_to_index = None
        self.item_index_to_id = None
        self.user_id_to_index = None
        self._load_resources()

    def _load_resources(self):
        """Loads models and mappings from MLflow or storage."""
        logger.info(f"Loading resources for stage: {self.model_stage}")
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

        # --- Load Mappings ---
        try:
            logger.info("Loading user and item maps...")
            # Assume maps are stored alongside processed data or logged as artifacts
            # Adjust paths based on where spark_job.py saves them
            self.user_map = pd.read_parquet(config['data_processing']['user_map_path'])
            self.item_map = pd.read_parquet(config['data_processing']['item_map_path'])
            self.user_id_to_index = pd.Series(self.user_map.user_index.values, index=self.user_map.user_id).to_dict()
            self.item_id_to_index = pd.Series(self.item_map.item_index.values, index=self.item_map.item_id).to_dict()
            self.item_index_to_id = pd.Series(self.item_map.item_id.values, index=self.item_map.item_index).to_dict()
            logger.info(f"Loaded {len(self.user_map)} users, {len(self.item_map)} items.")
        except Exception as e:
            logger.error(f"Failed to load user/item maps: {e}. Ensure paths are correct and accessible.", exc_info=True)
            # Service cannot operate without maps
            msg = "CRITICAL: Failed to load user/item maps. Service cannot start."
            logger.critical(msg)
            raise RuntimeError(msg) from e

        if self.user_map is None or self.item_map is None:
            msg = "CRITICAL: User/item maps are None after loading attempt. Service cannot start."
            logger.critical(msg)
            raise RuntimeError(msg)

        # --- Load CF Model (Spark ALS) ---
        try:
            cf_model_uri = f"models:/{self.cf_model_name}/{self.model_stage}"
            logger.info(f"Loading CF model from MLflow registry: {cf_model_uri}")
            # Loading Spark models via pyfunc is often easiest outside Spark env
            self.als_model = mlflow.pyfunc.load_model(cf_model_uri)
            logger.info("CF (ALS) model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load CF model '{self.cf_model_name}' (Stage: {self.model_stage}): {e}")
            # Fallback or raise error depending on requirements
            self.als_model = None

        # --- Load CB Model (TF-IDF Vectorizer & Matrix) ---
        try:
            cb_model_uri = f"models:/{self.cb_model_name}/{self.model_stage}"
            logger.info(f"Loading CB vectorizer from MLflow registry: {cb_model_uri}")
            self.tfidf_vectorizer = mlflow.sklearn.load_model(cb_model_uri)

            # Load the matrix separately (assuming it was saved and path logged)
            # Find the run associated with the model version to get the matrix path
            client = mlflow.tracking.MlflowClient()
            model_version_details = client.get_latest_versions(self.cb_model_name, stages=[self.model_stage])[0]
            run_info = client.get_run(model_version_details.run_id)
            matrix_path = run_info.data.params.get("tfidf_matrix_path", config['tfidf_model']['output_matrix_path']) # Fallback to config

            logger.info(f"Loading TF-IDF matrix from: {matrix_path}")
            # Requires cloud connector or local access
            self.item_tfidf_matrix = sp.load_npz(matrix_path)
            logger.info(f"CB (TF-IDF) vectorizer and matrix loaded successfully. Matrix shape: {self.item_tfidf_matrix.shape}")

        except Exception as e:
            logger.error(f"Failed to load CB model '{self.cb_model_name}' (Stage: {self.model_stage}): {e}", exc_info=True)
            self.tfidf_vectorizer = None
            self.item_tfidf_matrix = None

        # Check if at least one model loaded successfully
        if self.als_model is None and (self.tfidf_vectorizer is None or self.item_tfidf_matrix is None):
            msg = "CRITICAL: Neither CF (ALS) nor CB (TF-IDF) model components could be loaded. Service cannot provide recommendations."
            logger.critical(msg)
            raise RuntimeError(msg)
        elif self.als_model is None:
            logger.warning("CF (ALS) model failed to load. Service will rely solely on CB model if available.")
        elif self.tfidf_vectorizer is None or self.item_tfidf_matrix is None:
            logger.warning("CB (TF-IDF) model failed to load. Service will rely solely on CF model if available.")
        else:
            logger.info("Both CF and CB model components loaded successfully or their loading was attempted.")


    def get_cf_recommendations(self, user_id: int, k: int) -> list:
        """Get recommendations from the ALS model."""
        if self.als_model is None:
            logger.warning("ALS model not loaded, skipping CF recommendations.")
            return []
        if user_id not in self.user_id_to_index:
            logger.warning(f"User ID {user_id} not found in mapping, skipping CF.")
            return []

        user_index = self.user_id_to_index[user_id]
        try:
            # The Spark ALS model loaded via mlflow.pyfunc typically expects a Pandas DataFrame
            # with columns matching the feature names used during 'log_model' (e.g., 'user_index', 'item_index').
            # The current implementation simulates 'recommendForAllItems' for a specific user by
            # creating a DataFrame with all possible items for that user and then predicting ratings.
            # This can be inefficient for a large number of items.
            # If the MLflow PyFunc model was created with a custom wrapper that exposes a
            # 'recommendForUserSubset'-like capability, that would be more performant.
            # For now, we proceed with this approach, assuming the number of items is manageable
            # for this per-request scoring, or that the pyfunc model internally optimizes this.
            num_items = len(self.item_id_to_index)
            items_to_score = pd.DataFrame({
                'user_index': [user_index] * num_items,
                'item_index': list(range(num_items))
            })
            predictions = self.als_model.predict(items_to_score)
            # Sort predictions and get top K item indices
            top_k_indices = predictions.sort_values('prediction', ascending=False)['item_index'].head(k).tolist()
            top_k_ids = [self.item_index_to_id.get(idx, -1) for idx in top_k_indices if idx in self.item_index_to_id]
            return [item_id for item_id in top_k_ids if item_id != -1]

        except Exception as e:
            logger.error(f"Error getting CF recommendations for user {user_id}: {e}")
            return []

    def get_cb_recommendations(self, user_id: int, k: int, user_history: list = None) -> list:
        """Get recommendations based on TF-IDF cosine similarity."""
        if self.tfidf_vectorizer is None or self.item_tfidf_matrix is None:
            logger.warning("TF-IDF model/matrix not loaded, skipping CB recommendations.")
            return []
        if not user_history:
             # If no history provided, maybe fetch from a DB or return popular?
             # For now, return empty if no history given for CB.
            logger.warning(f"No interaction history provided for user {user_id}, skipping CB.")
            return []

        # Convert history item IDs to indices
        history_indices = [self.item_id_to_index.get(item_id) for item_id in user_history if item_id in self.item_id_to_index]
        if not history_indices:
            logger.warning(f"User {user_id} history contains no known items.")
            return []

        # Calculate user profile vector (average TF-IDF of history)
        user_profile_vector = self.item_tfidf_matrix[history_indices].mean(axis=0)
        # Ensure it's a 2D array for cosine_similarity
        user_profile_vector = np.asarray(user_profile_vector)
        if user_profile_vector.ndim == 1:
             user_profile_vector = user_profile_vector.reshape(1, -1)


        # Calculate cosine similarity
        similarities = cosine_similarity(user_profile_vector, self.item_tfidf_matrix)[0] # Get the single row of similarities

        # Get top K *excluding* items already in history
        # Create pairs of (similarity, item_index)
        sim_scores = list(enumerate(similarities))
        # Sort by similarity (descending)
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Filter out history items and get top K
        recommended_indices = []
        history_indices_set = set(history_indices)
        for index, score in sim_scores:
            if index not in history_indices_set:
                recommended_indices.append(index)
            if len(recommended_indices) >= k:
                break

        top_k_ids = [self.item_index_to_id.get(idx, -1) for idx in recommended_indices if idx in self.item_index_to_id]
        return [item_id for item_id in top_k_ids if item_id != -1]


    def recommend(self, user_id: int, k: int, user_history: list = None) -> list:
        """Generates hybrid recommendations."""
        logger.info(f"Generating recommendations for user_id: {user_id}")

        cf_recs = self.get_cf_recommendations(user_id, k)
        cb_recs = self.get_cb_recommendations(user_id, k, user_history)

        # Simple Hybrid: Combine and deduplicate, maybe prioritize CF?
        combined_recs = cf_recs + [rec for rec in cb_recs if rec not in cf_recs]
        final_recs = combined_recs[:k]

        logger.info(f"Generated {len(final_recs)} recommendations for user {user_id}.")
        return final_recs

# Global instance (lazy loaded on first request in FastAPI)
recommendation_model_instance = None

def get_recommendation_model():
    global recommendation_model_instance
    if recommendation_model_instance is None:
        logger.info("Initializing RecommendationModel instance...")
        recommendation_model_instance = RecommendationModel(
            cf_model_name=os.getenv("MODEL_NAME_CF", config['mlflow_config']['cf_model_name']),
            cb_model_name=os.getenv("MODEL_NAME_CB", config['mlflow_config']['cb_model_name']),
            model_stage=os.getenv("MODEL_STAGE", "Production")
        )
        logger.info("RecommendationModel instance initialized.")
    return recommendation_model_instance