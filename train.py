#!/usr/bin/env python
# coding: utf-8

import os
import sys  # Add this import statement
import numpy as np
import pandas as pd

import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')  # Comment out or remove this line

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import euclidean_distances
from scipy.spatial.distance import cdist

import warnings
warnings.filterwarnings("ignore")

import subprocess
subprocess.check_call([sys.executable, "-m", "pip", "install", "spotipy"])


# In[49]:


data = pd.read_csv("dataset/data.csv")
genre_data = pd.read_csv('dataset/data_by_genres.csv')
year_data = pd.read_csv('dataset/data_by_year.csv')


# In[50]:


print(data.info())


# In[51]:


print(genre_data.info())


# In[52]:


print(year_data.info())


# In[53]:


from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Assuming genre_data is a DataFrame and np has been imported as numpy

# Create a pipeline and fit it in one go
cluster_pipeline = make_pipeline(StandardScaler(), KMeans(n_clusters=10))
genre_data['cluster'] = cluster_pipeline.fit_predict(genre_data.select_dtypes(include=[np.number]))


# In[54]:


from sklearn.manifold import TSNE
from sklearn.manifold import TSNE
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd
import plotly.express as px

# Create a pipeline and apply t-SNE transformation in one step
tsne_pipeline = make_pipeline(StandardScaler(), TSNE(n_components=2, verbose=1, random_state=42))
projection = pd.DataFrame(tsne_pipeline.fit_transform(genre_data.select_dtypes(include=[np.number])),
                          columns=['x', 'y'])
projection['genres'] = genre_data['genres']
projection['cluster'] = genre_data['cluster']

# Create the scatter plot
fig = px.scatter(projection, x='x', y='y', color='cluster', hover_data=['genres'])
fig.show()


# In[55]:


song_cluster_pipeline = Pipeline([('scaler', StandardScaler()),
                                  ('kmeans', KMeans(n_clusters=20,
                                   verbose=False))
                                 ], verbose=False)

X = data.select_dtypes(np.number)
number_cols = list(X.columns)
song_cluster_pipeline.fit(X)
song_cluster_labels = song_cluster_pipeline.predict(X)
data['cluster_label'] = song_cluster_labels


# In[56]:


from sklearn.decomposition import PCA

pca_pipeline = Pipeline([('scaler', StandardScaler()), ('PCA', PCA(n_components=2))])
song_embedding = pca_pipeline.fit_transform(X)
projection = pd.DataFrame(columns=['x', 'y'], data=song_embedding)
projection['title'] = data['name']
projection['cluster'] = data['cluster_label']

fig = px.scatter(
    projection, x='x', y='y', color='cluster', hover_data=['x', 'y', 'title'])
fig.show()


# In[57]:


import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.exceptions import SpotifyException  # Import SpotifyException
from collections import defaultdict

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id='c7461b2661ef46ebaec1f96ae329dc36',
                                                           client_secret='ddb8161e4b1046d78bd175a74f032136'))



def find_song(name, year):
    song_data = defaultdict()
    try:
        results = sp.search(q= 'track: {} year: {}'.format(name, year), limit=1)
        if results['tracks']['items'] == []:
            return None

        results = results['tracks']['items'][0]
        track_id = results['id']
        audio_features = sp.audio_features(track_id)[0]

        song_data['name'] = [name]
        song_data['year'] = [year]
        song_data['explicit'] = [int(results['explicit'])]
        song_data['duration_ms'] = [results['duration_ms']]
        song_data['popularity'] = [results['popularity']]

        for key, value in audio_features.items():
            song_data[key] = value

        return pd.DataFrame(song_data)
    except SpotifyException as e:
        print(f"Spotify API error: {e}")
        return None


# In[59]:


from collections import defaultdict
from sklearn.metrics import euclidean_distances
from scipy.spatial.distance import cdist
import difflib

number_cols = ['valence', 'year', 'acousticness', 'danceability', 'duration_ms', 'energy', 'explicit',
 'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'popularity', 'speechiness', 'tempo']


def get_song_data(song, spotify_data):

    try:
        song_data = spotify_data[(spotify_data['name'] == song['name'])
                                & (spotify_data['year'] == song['year'])]
        if song_data.empty:
            raise IndexError
        return song_data.iloc[0]

    except IndexError:
        return find_song(song['name'], song['year'])


def get_mean_vector(song_list, spotify_data):

    song_vectors = []

    for song in song_list:
        song_data = get_song_data(song, spotify_data)
        if song_data is None:
            print('Warning: {} does not exist in Spotify or in database'.format(song['name']))
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
    # First, get the common columns between spotify_data and genre_data
    common_cols = list(set(spotify_data.columns) & set(genre_data.columns))
    # Merge on common audio features
    merged_data = pd.merge(spotify_data, genre_data, 
                          on=common_cols,
                          how='inner')
    song_center = get_mean_vector(song_list, spotify_data)  # Use spotify_data for center
    scaler = song_cluster_pipeline.steps[0][1]
    if vibe:
        if vibe == 'happy':
            merged_data = merged_data[merged_data['danceability'] > 0.5]
        elif vibe == 'sad':
            merged_data = merged_data[merged_data['danceability'] <= 0.5]
        if merged_data.empty:
            print("No songs found after applying vibe filter.")
            return []
    if genre and 'genres' in merged_data.columns:
        print(f"Filtering by genre: {genre}")
        merged_data = merged_data[merged_data['genres'].str.contains(genre, case=False, na=False)]
        print(f"Number of songs after genre filter: {len(merged_data)}")
        if merged_data.empty:
            print("No songs found after applying genre filter.")
            return []
    number_cols = [
        'valence', 'year', 'acousticness', 'danceability', 'duration_ms',
        'energy', 'explicit', 'instrumentalness', 'key', 'liveness', 
        'loudness', 'mode', 'popularity', 'speechiness', 'tempo'
    ]
    number_cols = [col for col in number_cols if col in merged_data.columns]
    print(f"Using features: {number_cols}")
    scaled_data = scaler.transform(merged_data[number_cols])
    scaled_song_center = scaler.transform(song_center.reshape(1, -1))
    distances = cdist(scaled_song_center, scaled_data, 'cosine')
    index = list(np.argsort(distances)[:, :n_songs][0])
    rec_songs = merged_data.iloc[index]
    rec_songs = rec_songs[~rec_songs['name'].isin(song_dict['name'])]
    # Ensure we only return columns that exist
    available_metadata = [col for col in metadata_cols if col in rec_songs.columns]
    return rec_songs[available_metadata].to_dict(orient='records')

def prepare_data(spotify_data, genre_data):
    common_numeric = [
        'acousticness', 'danceability', 'duration_ms',
        'energy', 'instrumentalness', 'liveness',
        'loudness', 'speechiness', 'tempo', 'key', 'mode'
    ]
    # Convert only common columns to numeric
    for col in common_numeric:
        if col in spotify_data.columns:
            spotify_data[col] = pd.to_numeric(spotify_data[col], errors='coerce')
        if col in genre_data.columns:
            genre_data[col] = pd.to_numeric(genre_data[col], errors='coerce')
    return spotify_data, genre_data

# Test the function
test_song_list = [
    {'name': 'Stay Away', 'year': 1993},
    {'name': 'Come As You Are', 'year': 1991},
    {'name': 'Smells Like Teen Spirit', 'year': 1991},
    {'name': 'Lithium', 'year': 1992},
    {'name': 'Perfect', 'year': 2017}
]

# Prepare the data
data, genre_data = prepare_data(data, genre_data)

# Print column names for debugging
print("Spotify data columns:", data.columns.tolist())
print("Genre data columns:", genre_data.columns.tolist())

vibe = 'happy'
genre = 'house'
recommendations = recommend_songs(test_song_list, data, genre_data, vibe=vibe, genre=genre)
print(recommendations)

