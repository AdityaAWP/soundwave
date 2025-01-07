from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
from train import prepare_data, recommend_songs, data, genre_data

app = Flask(__name__)
CORS(app)  # Enable CORS

# Prepare the data
data, genre_data = prepare_data(data, genre_data)

@app.route('/recommend', methods=['POST'])
def recommend():
    content = request.json
    song_list = content.get('songs', [])
    genre = content.get('genre', None)
    vibe = content.get('vibe', None)
    
    recommendations = recommend_songs(song_list, data, genre_data, genre=genre, vibe=vibe)
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True, port=5001)  # Change the port to 5001 or any other available port
