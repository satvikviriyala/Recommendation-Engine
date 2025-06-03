from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType
from pyspark.ml.recommendation import ALS
from pyspark.ml.feature import StringIndexer, IndexToString # For mapping
from pyspark.ml import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import numpy as np
import scipy.sparse as sp
import os
import mlflow
import mlflow.spark
import mlflow.sklearn

# Load utils relative to project root if running spark-submit
import sys
# Add project root to path - adjust if needed based on execution context
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.utils.config_loader import load_config
from src.utils.logging_utils import setup_logger

config = load_config()
proc_cfg = config['data_processing']
als_cfg = config['als_model']
tfidf_cfg = config['tfidf_model']
mlflow_cfg = config['mlflow_config']
spark_master = os.getenv("SPARK_MASTER", "local[*]") # Get from env or default

logger = setup_logger("SparkProcessingJob")

def process_data_and_train(spark: SparkSession):
    """Loads data, processes it, trains ALS and TF-IDF, logs to MLflow."""
    logger.info("Starting Spark job...")
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment(mlflow_cfg['experiment_name'])

    # --- Load Raw Data ---
    logger.info(f"Loading raw interactions from {proc_cfg['raw_interactions_path']}")
    interactions_df = spark.read.parquet(proc_cfg['raw_interactions_path'])
    logger.info(f"Loading raw items from {proc_cfg['raw_items_path']}")
    items_df = spark.read.parquet(proc_cfg['raw_items_path'])

    # --- Feature Engineering & Mapping ---
    logger.info("Indexing users and items...")
    # Create unique integer IDs for users and items
    user_indexer = StringIndexer(inputCol="user_id", outputCol="user_index", handleInvalid="skip")
    item_indexer = StringIndexer(inputCol="item_id", outputCol="item_index", handleInvalid="skip")

    pipeline = Pipeline(stages=[user_indexer, item_indexer])
    mapping_model = pipeline.fit(interactions_df)
    interactions_indexed = mapping_model.transform(interactions_df)

    # Save mappings (important for inference)
    user_map = mapping_model.stages[0].labels
    item_map = mapping_model.stages[1].labels
    # Convert to Pandas for easier saving/loading outside Spark
    user_map_pd = pd.DataFrame(user_map, columns=['user_id'])
    item_map_pd = pd.DataFrame(item_map, columns=['item_id'])
    user_map_pd['user_index'] = user_map_pd.index
    item_map_pd['item_index'] = item_map_pd.index

    logger.info(f"Saving user map ({len(user_map_pd)} users) to {proc_cfg['user_map_path']}")
    user_map_pd.to_parquet(proc_cfg['user_map_path'], index=False) # Requires cloud connector setup
    logger.info(f"Saving item map ({len(item_map_pd)} items) to {proc_cfg['item_map_path']}")
    item_map_pd.to_parquet(proc_cfg['item_map_path'], index=False)

    # Cast columns for ALS
    interactions_final = interactions_indexed \
        .withColumn("user_index", F.col("user_index").cast(IntegerType())) \
        .withColumn("item_index", F.col("item_index").cast(IntegerType())) \
        .withColumn("rating", F.col("rating").cast("float")) \
        .select("user_index", "item_index", "rating")

    # --- Train Spark ALS Model (Collaborative Filtering) ---
    logger.info("Training Spark ALS model...")
    als = ALS(
        rank=als_cfg['rank'],
        maxIter=als_cfg['maxIter'],
        regParam=als_cfg['regParam'],
        userCol="user_index",
        itemCol="item_index",
        ratingCol="rating",
        coldStartStrategy=als_cfg['coldStartStrategy'],
        implicitPrefs=False # Set to True if using interaction counts instead of ratings
    )

    with mlflow.start_run(run_name="Spark_ALS_Training") as als_run:
        mlflow.log_params({
            "als_rank": als_cfg['rank'],
            "als_maxIter": als_cfg['maxIter'],
            "als_regParam": als_cfg['regParam'],
            "als_coldStartStrategy": als_cfg['coldStartStrategy']
        })

        als_model = als.fit(interactions_final)

        # Log model to MLflow
        logger.info(f"Logging Spark ALS model '{mlflow_cfg['cf_model_name']}' to MLflow...")
        mlflow.spark.log_model(
            als_model,
            artifact_path="spark-als-model", # Subdirectory within MLflow run artifacts
            registered_model_name=mlflow_cfg['cf_model_name']
        )
        logger.info("ALS Model logged.")

        # (Optional) Evaluate ALS - Requires splitting data earlier
        # predictions = als_model.transform(test_data)
        # evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
        # rmse = evaluator.evaluate(predictions)
        # logger.info(f"ALS RMSE on test data: {rmse}")
        # mlflow.log_metric("als_rmse", rmse)

        # Save factors separately if needed for direct loading in serving
        # user_factors = als_model.userFactors
        # item_factors = als_model.itemFactors
        # user_factors.write.parquet(os.path.join(als_cfg['output_path'], "user_factors.parquet"), mode="overwrite")
        # item_factors.write.parquet(os.path.join(als_cfg['output_path'], "item_factors.parquet"), mode="overwrite")
        # logger.info("ALS factors saved separately.")

    # --- Calculate TF-IDF Features (Content-Based) ---
    # This part might be better done outside Spark if item metadata isn't huge,
    # as sklearn is often easier for TF-IDF. Assumes items_df fits in driver memory.
    logger.info("Calculating TF-IDF features for items...")
    items_pd = items_df.toPandas() # Collect item data to driver
    items_pd = items_pd.merge(item_map_pd, on='item_id', how='inner') # Add item_index
    items_pd = items_pd.sort_values('item_index').reset_index(drop=True) # Ensure order matches index

    # Use description or combine text features
    corpus = items_pd['description'].fillna('') # Handle potential missing descriptions

    vectorizer = TfidfVectorizer(
        max_features=tfidf_cfg['max_features'],
        min_df=tfidf_cfg['min_df'],
        stop_words='english'
    )
    item_tfidf_matrix = vectorizer.fit_transform(corpus)

    logger.info(f"TF-IDF Matrix shape: {item_tfidf_matrix.shape}")

    # Save TF-IDF vectorizer and matrix
    logger.info(f"Saving TF-IDF vectorizer to {tfidf_cfg['output_vectorizer_path']}")
    joblib.dump(vectorizer, tfidf_cfg['output_vectorizer_path']) # Requires cloud connector
    logger.info(f"Saving TF-IDF matrix to {tfidf_cfg['output_matrix_path']}")
    sp.save_npz(tfidf_cfg['output_matrix_path'], item_tfidf_matrix) # Requires cloud connector

    # Log TF-IDF model to MLflow (important for serving consistency)
    with mlflow.start_run(run_name="Sklearn_TFIDF_Calculation") as tfidf_run:
         mlflow.log_params({
             "tfidf_max_features": tfidf_cfg['max_features'],
             "tfidf_min_df": tfidf_cfg['min_df']
         })
         logger.info(f"Logging TF-IDF vectorizer '{mlflow_cfg['cb_model_name']}' to MLflow...")
         mlflow.sklearn.log_model(
             vectorizer,
             artifact_path="sklearn-tfidf-vectorizer",
             registered_model_name=mlflow_cfg['cb_model_name']
         )
         # Log the matrix path as a parameter or artifact for reference
         mlflow.log_param("tfidf_matrix_path", tfidf_cfg['output_matrix_path'])
         # Or log the matrix itself if not too large
         # mlflow.log_artifact(tfidf_cfg['output_matrix_path'])
         logger.info("TF-IDF Vectorizer logged.")


    logger.info("Spark job finished successfully.")


if __name__ == "__main__":
    spark = SparkSession.builder \
        .appName(proc_cfg['spark_app_name']) \
        .master(spark_master) \
        .config("spark.sql.parquet.writeLegacyFormat", "true") \
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
        .config("spark.hadoop.fs.gs.impl", "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem") \
        .config("spark.hadoop.google.cloud.auth.service.account.enable", "true") \
        .config("spark.hadoop.google.cloud.auth.service.account.json.keyfile", os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")) \
        .getOrCreate()

    # Configure AWS credentials if using S3 (better via instance profile/env vars)
    # sc = spark.sparkContext
    # sc._jsc.hadoopConfiguration().set("fs.s3a.access.key", os.getenv("AWS_ACCESS_KEY_ID", ""))
    # sc._jsc.hadoopConfiguration().set("fs.s3a.secret.key", os.getenv("AWS_SECRET_ACCESS_KEY", ""))

    process_data_and_train(spark)
    spark.stop()