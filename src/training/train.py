import os
import yaml
import mlflow
import logging
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import joblib
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config():
    """Load configuration from yaml file."""
    config_path = Path("configs/config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)

def init_spark():
    """Initialize Spark session."""
    return (SparkSession.builder
            .appName("RecSysTraining")
            .config("spark.executor.memory", "4g")
            .config("spark.driver.memory", "2g")
            .getOrCreate())

def train_collaborative_filtering(spark, config, interactions_df):
    """Train ALS collaborative filtering model."""
    logger.info("Training ALS model...")
    
    als = ALS(
        rank=config['als_model']['rank'],
        maxIter=config['als_model']['maxIter'],
        regParam=config['als_model']['regParam'],
        coldStartStrategy=config['als_model']['coldStartStrategy'],
        userCol="user_id",
        itemCol="item_id",
        ratingCol="rating"
    )
    
    model = als.fit(interactions_df)
    
    # Save model factors
    output_path = os.path.expandvars(config['als_model']['output_path'])
    model.save(output_path)
    logger.info(f"ALS model saved to {output_path}")
    
    return model

def train_content_based(items_df, config):
    """Train TF-IDF content-based model."""
    logger.info("Training TF-IDF model...")
    
    # Combine text features
    items_df['text_features'] = items_df['title'] + ' ' + items_df['description']
    
    vectorizer = TfidfVectorizer(
        max_features=config['tfidf_model']['max_features'],
        min_df=config['tfidf_model']['min_df']
    )
    
    tfidf_matrix = vectorizer.fit_transform(items_df['text_features'])
    
    # Save vectorizer and matrix
    output_vectorizer_path = os.path.expandvars(config['tfidf_model']['output_vectorizer_path'])
    output_matrix_path = os.path.expandvars(config['tfidf_model']['output_matrix_path'])
    
    joblib.dump(vectorizer, output_vectorizer_path)
    np.savez(output_matrix_path, data=tfidf_matrix.data, indices=tfidf_matrix.indices,
             indptr=tfidf_matrix.indptr, shape=tfidf_matrix.shape)
    
    logger.info(f"TF-IDF model saved to {output_vectorizer_path} and {output_matrix_path}")
    
    return vectorizer, tfidf_matrix

def main():
    """Main training pipeline."""
    # Load environment variables and config
    load_dotenv()
    config = load_config()
    
    # Initialize MLflow
    mlflow.set_experiment(config['mlflow_config']['experiment_name'])
    
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params({
            'als_rank': config['als_model']['rank'],
            'als_max_iter': config['als_model']['maxIter'],
            'als_reg_param': config['als_model']['regParam'],
            'tfidf_max_features': config['tfidf_model']['max_features'],
            'tfidf_min_df': config['tfidf_model']['min_df']
        })
        
        # Initialize Spark
        spark = init_spark()
        
        try:
            # Load data
            interactions_df = spark.read.parquet(
                os.path.expandvars(config['data_processing']['raw_interactions_path'])
            )
            items_df = spark.read.parquet(
                os.path.expandvars(config['data_processing']['raw_items_path'])
            )
            
            # Train models
            cf_model = train_collaborative_filtering(spark, config, interactions_df)
            cb_vectorizer, cb_matrix = train_content_based(items_df.toPandas(), config)
            
            # Log models to MLflow
            mlflow.spark.log_model(cf_model, config['mlflow_config']['cf_model_name'])
            mlflow.sklearn.log_model(cb_vectorizer, config['mlflow_config']['cb_model_name'])
            
            logger.info("Training completed successfully!")
            
        finally:
            spark.stop()

if __name__ == "__main__":
    main() 