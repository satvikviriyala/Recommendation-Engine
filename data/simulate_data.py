# To run this script directly from the project root, ensure the root directory is in your PYTHONPATH:
# export PYTHONPATH=$PYTHONPATH:.
# Alternatively, run as a module: python -m data.simulate_data

import pandas as pd
import numpy as np
import os
import random
from sklearn.feature_extraction.text import TfidfVectorizer # For realistic text
from src.utils.config_loader import load_config

config = load_config()
sim_cfg = config['simulation']
proc_cfg = config['data_processing']

logger = setup_logger("DataSimulator")

def generate_realistic_text(num_samples, vocab_size=500, max_len=50):
    """Generates somewhat realistic text snippets."""
    # Create a dummy corpus and vectorizer to get plausible words
    dummy_corpus = [" ".join([f"word{random.randint(0, vocab_size)}" for _ in range(random.randint(10, 100))]) for _ in range(100)]
    vectorizer = TfidfVectorizer(max_features=vocab_size)
    vectorizer.fit(dummy_corpus)
    vocab = vectorizer.get_feature_names_out()

    texts = []
    for _ in range(num_samples):
        length = random.randint(5, max_len)
        texts.append(" ".join(random.choices(vocab, k=length)))
    return texts

def simulate_data():
    """Generates simulated interaction and item data."""
    logger.info(f"Starting data simulation: {sim_cfg['num_users']} users, {sim_cfg['num_items']} items, {sim_cfg['num_interactions']} interactions.")

    # Interactions
    user_ids = np.random.randint(1, sim_cfg['num_users'] + 1, sim_cfg['num_interactions'])
    item_ids = np.random.randint(1, sim_cfg['num_items'] + 1, sim_cfg['num_interactions'])
    ratings = np.random.choice([1, 2, 3, 4, 5], sim_cfg['num_interactions'], p=[0.1, 0.1, 0.2, 0.3, 0.3]) # Skewed towards positive
    timestamps = pd.to_datetime(np.random.randint(1640995200, 1672531199, sim_cfg['num_interactions']), unit='s') # Approx 2022

    interactions_df = pd.DataFrame({
        'user_id': user_ids,
        'item_id': item_ids,
        'rating': ratings,
        'timestamp': timestamps
    })
    # Ensure unique user-item pairs if needed, or allow multiple interactions
    interactions_df = interactions_df.drop_duplicates(subset=['user_id', 'item_id'], keep='last')
    logger.info(f"Generated {len(interactions_df)} unique interactions.")

    # Items
    item_ids_unique = np.arange(1, sim_cfg['num_items'] + 1)
    descriptions = generate_realistic_text(sim_cfg['num_items'])
    genres = [random.choice(['Action|Adventure', 'Comedy', 'Drama|Romance', 'Sci-Fi|Thriller', 'Documentary', 'Animation|Children']) for _ in range(sim_cfg['num_items'])]

    items_df = pd.DataFrame({
        'item_id': item_ids_unique,
        'description': descriptions,
        'genres': genres
    })
    logger.info(f"Generated {len(items_df)} items.")

    # Save locally (or directly upload to cloud storage if preferred)
    output_dir = sim_cfg['output_path']
    os.makedirs(output_dir, exist_ok=True)
    interactions_path = os.path.join(output_dir, "interactions.parquet")
    items_path = os.path.join(output_dir, "items.parquet")

    interactions_df.to_parquet(interactions_path, index=False)
    items_df.to_parquet(items_path, index=False)
    logger.info(f"Saved simulated data to {output_dir}")
    logger.info(f"--- IMPORTANT: Upload these files to your raw data bucket ---")
    logger.info(f"Interactions -> {proc_cfg['raw_interactions_path']}")
    logger.info(f"Items -> {proc_cfg['raw_items_path']}")

if __name__ == "__main__":
    simulate_data()