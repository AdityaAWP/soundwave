from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from werkzeug.security import generate_password_hash, check_password_hash
from train import prepare_data, recommend_songs, data, genre_data

app = Flask(__name__)
CORS(app)  # Add this line to enable CORS

# Configure MySQL database
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:@localhost/soundwave'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
migrate = Migrate(app, db)  # Initialize Flask-Migrate

# Define User and Playlist models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    playlists = db.relationship('Playlist', backref='user', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Playlist(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    song_name = db.Column(db.String(120), nullable=False)
    song_artist = db.Column(db.String(120), nullable=False)
    song_year = db.Column(db.Integer, nullable=False)

# Create the database tables
with app.app_context():
    db.create_all()

# Prepare the data
data, genre_data = prepare_data(data, genre_data)

@app.route('/register', methods=['POST'])
def register():
    content = request.json
    username = content.get('username')
    password = content.get('password')

    if User.query.filter_by(username=username).first():
        return jsonify({'message': 'Username already exists'}), 400

    user = User(username=username)
    user.set_password(password)
    db.session.add(user)
    db.session.commit()

    return jsonify({'message': 'User registered successfully'})

@app.route('/login', methods=['POST'])
def login():
    content = request.json
    username = content.get('username')
    password = content.get('password')

    user = User.query.filter_by(username=username).first()
    if user is None or not user.check_password(password):
        return jsonify({'message': 'Invalid username or password'}), 401

    return jsonify({'message': 'Login successful'})

@app.route('/recommend', methods=['POST'])
def recommend():
    content = request.json
    song_list = content.get('songs', [])
    genre = content.get('genre', None)
    vibe = content.get('vibe', None)
    
    recommendations = recommend_songs(song_list, data, genre_data, genre=genre, vibe=vibe)
    return jsonify(recommendations)

@app.route('/like', methods=['POST'])
def like_song():
    content = request.json
    username = content.get('username')
    song_name = content.get('song_name')
    song_artist = content.get('song_artist')
    song_year = content.get('song_year')

    user = User.query.filter_by(username=username).first()
    if not user:
        return jsonify({'message': 'User not found'}), 404

    playlist = Playlist(user_id=user.id, song_name=song_name, song_artist=song_artist, song_year=song_year)
    db.session.add(playlist)
    db.session.commit()

    return jsonify({'message': 'Song liked successfully'})

@app.route('/playlist/<username>', methods=['GET'])
def get_playlist(username):
    user = User.query.filter_by(username=username).first()
    if not user:
        return jsonify({'message': 'User not found'}), 404

    playlist = Playlist.query.filter_by(user_id=user.id).all()
    return jsonify([{'song_name': p.song_name, 'song_artist': p.song_artist, 'song_year': p.song_year} for p in playlist])

if __name__ == '__main__':
    app.run(debug=True, port=5001)