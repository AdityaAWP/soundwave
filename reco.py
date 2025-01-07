import os
import numpy as np
import pandas as pd
import logging
from flask import Flask, request, jsonify
from pydantic import BaseModel
from typing import List, Optional
from collections import defaultdict
from sklearn.metrics import euclidean_distances
from scipy.spatial.distance import cdist
import joblib
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from logging.handlers import RotatingFileHandler
from flask_cors import CORS
from blueprints.frontend import frontend

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add a rotating file handler
handler = RotatingFileHandler('app.log', maxBytes=2000, backupCount=10)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

logger.info("Loading model and data...")
song_cluster_pipeline = joblib.load('models/song_cluster_pipeline.pkl')
data = pd.read_csv('dataset/data_sampled.csv')
genre_data = pd.read_csv('dataset/genre_data_sampled.csv')
logger.info("Model and data loaded successfully")

app = Flask(__name__)
CORS(app)

# Enable debug mode
app.debug = True

# Register the frontend blueprint
app.register_blueprint(frontend)

# Add a test route to verify the server is working
@app.route('/api/test')
def test():
    return jsonify({"status": "Server is running"})

number_cols = ['valence', 'year', 'acousticness', 'danceability', 'duration_ms', 'energy', 'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'popularity', 'speechiness', 'tempo']

def get_song_data(song, spotify_data):
    try:
        song_data = spotify_data[(spotify_data['name'] == song['name'])
                                & (spotify_data['year'] == song['year'])].iloc[0]
        return song_data
    except IndexError:
        return None

def get_mean_vector(song_list, spotify_data):
    song_vectors = []
    for song in song_list:
        song_data = get_song_data(song, spotify_data)
        if song_data is None:
            logger.warning(f"Warning: {song['name']} does not exist in Spotify or in database")
            continue
        song_vector = song_data[number_cols].values
        song_vectors.append(song_vector)
    song_matrix = np.array(list(song_vectors))
    return np.mean(song_matrix, axis=0)

def flatten_dict_list(dict_list):
    flattened_dict = defaultdict()
    for key in dict_list[0].keys():
        flattened_dict[key] = []
    for dictionary in dict_list:
        for key, value in dictionary.items():
            flattened_dict[key].append(value)
    return flattened_dict

def recommend_songs(song_list=None, spotify_data=None, genre_data=None, n_songs=10, vibe=None, genre=None):
    metadata_cols = ['name', 'year', 'artists']
    song_dict = flatten_dict_list(song_list)
    common_cols = list(set(spotify_data.columns) & set(genre_data.columns))
    merged_data = pd.merge(spotify_data, genre_data, on=common_cols, how='inner')
    song_center = get_mean_vector(song_list, spotify_data)
    scaler = song_cluster_pipeline.steps[0][1]
    if vibe:
        if vibe == 'happy':
            merged_data = merged_data[merged_data['danceability'] > 0.5]
        elif vibe == 'sad':
            merged_data = merged_data[merged_data['danceability'] <= 0.5]
    if genre and 'genres' in merged_data.columns:
        logger.info(f"Filtering by genre: {genre}")
        merged_data = merged_data[merged_data['genres'].str.contains(genre, case=False, na=False)]
        logger.info(f"Number of songs after genre filter: {len(merged_data)}")
    number_cols = [col for col in number_cols if col in merged_data.columns]
    logger.info(f"Using features: {number_cols}")
    scaled_data = scaler.transform(merged_data[number_cols])
    scaled_song_center = scaler.transform(song_center.reshape(1, -1))
    distances = cdist(scaled_song_center, scaled_data, 'cosine')
    index = list(np.argsort(distances)[:, :n_songs][0])
    rec_songs = merged_data.iloc[index]
    rec_songs = rec_songs[~rec_songs['name'].isin(song_dict['name'])]
    available_metadata = [col for col in metadata_cols if col in rec_songs.columns]
    return rec_songs[available_metadata].to_dict(orient='records')

executor = ThreadPoolExecutor(max_workers=4)

@lru_cache(maxsize=128)
def recommend_songs_cached(song_list_str, n_songs, vibe, genre):
    song_list = eval(song_list_str)  # Convert string back to list
    return recommend_songs(song_list, data, genre_data, n_songs=n_songs, vibe=vibe, genre=genre)

@app.route("/recommend_songs", methods=["POST"])
def recommend_songs_api():
    start_time = time.time()
    try:
        request_data = request.get_json()
        logger.info("Received request: %s", request_data)
        
        if not request_data or 'song_list' not in request_data:
            return jsonify({"detail": "Invalid request data"}), 400
            
        song_list = [{'name': song['name'], 'year': song['year']} 
                    for song in request_data['song_list'] 
                    if 'name' in song and 'year' in song]
        
        if not song_list:
            return jsonify({"detail": "No valid songs in request"}), 400
            
        song_list_str = str(song_list)
        logger.info("Song list: %s", song_list)
        
        recommendations = recommend_songs_cached(
            song_list_str, 
            request_data.get('n_songs', 10),
            request_data.get('vibe'),
            request_data.get('genre')
        )
        
        if not recommendations:
            return jsonify([]), 200  # Return empty array instead of 404
            
        logger.info("Recommendations: %s", recommendations)
        end_time = time.time()
        logger.info(f"Request processed in {end_time - start_time} seconds")
        
        response = jsonify(recommendations)
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
        
    except Exception as e:
        logger.error("Error processing request: %s", e)
        return jsonify({"detail": str(e)}), 500

@app.route("/coba", methods=["GET"])
def coba():
    return jsonify({"message": "Hello World"})

if __name__ == "__main__":
    from waitress import serve
    logger.info("Starting server on http://localhost:8000")
    app.run(host="127.0.0.1", port=8000, debug=True)  # Use app.run instead of waitress for debugging