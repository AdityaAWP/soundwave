import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [songs, setSongs] = useState([{ name: '', year: '' }]);
  const [genre, setGenre] = useState('');
  const [vibe, setVibe] = useState('');
  const [recommendations, setRecommendations] = useState([]);

  const handleSongChange = (index, event) => {
    const newSongs = [...songs];
    newSongs[index][event.target.name] = event.target.value;
    setSongs(newSongs);
  };

  const addSong = () => {
    setSongs([...songs, { name: '', year: '' }]);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    const response = await axios.post('http://localhost:5001/recommend', {  // Use port 5001
      songs,
      genre,
      vibe,
    });
    setRecommendations(response.data);
  };

  return (
    <div className="App">
      <h1>Music Recommendation</h1>
      <form onSubmit={handleSubmit}>
        {songs.map((song, index) => (
          <div key={index}>
            <input
              type="text"
              name="name"
              placeholder="Song Name"
              value={song.name}
              onChange={(event) => handleSongChange(index, event)}
            />
            <input
              type="text"
              name="year"
              placeholder="Year"
              value={song.year}
              onChange={(event) => handleSongChange(index, event)}
            />
          </div>
        ))}
        <button type="button" onClick={addSong}>Add Song</button>
        <input
          type="text"
          placeholder="Genre"
          value={genre}
          onChange={(event) => setGenre(event.target.value)}
        />
        <input
          type="text"
          placeholder="Vibe"
          value={vibe}
          onChange={(event) => setVibe(event.target.value)}
        />
        <button type="submit">Get Recommendations</button>
      </form>
      <h2>Recommendations</h2>
      <ul>
        {recommendations.map((rec, index) => (
          <li key={index}>{rec.name} by {rec.artists} ({rec.year})</li>
        ))}
      </ul>
    </div>
  );
}

export default App;
