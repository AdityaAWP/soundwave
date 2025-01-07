import React, { useState } from "react";
import JumbotronImage from "../assets/jumbotron.jpg";
import axios from 'axios';
import { ClipLoader } from 'react-spinners';
import { useNavigate } from 'react-router-dom';

const Recomendation = () => {
    const [songs, setSongs] = useState([{ name: '', year: '' }]);
    const [genre, setGenre] = useState('');
    const [vibe, setVibe] = useState('');
    const [recommendations, setRecommendations] = useState([]);
    const [coverUrls, setCoverUrls] = useState({});
    const [loading, setLoading] = useState(false);
    const [likedSongs, setLikedSongs] = useState([]);
    const navigate = useNavigate();

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
        setLoading(true);
        const response = await axios.post('http://localhost:5001/recommend', {
            songs,
            genre,
            vibe,
        });
        setRecommendations(response.data);
        await fetchCoverUrls(response.data);
        setLoading(false);
    };

    const fetchCoverUrls = async (recommendations) => {
        const urls = {};
        for (const rec of recommendations) {
            try {
                const response = await axios.get(`https://api.spotify.com/v1/search?q=track:${rec.name}%20artist:${rec.artists}&type=track`, {
                    headers: {
                        'Authorization': `Bearer c6a9d98b65e58a344754d6ccbe44811fc1b45fef9334d6e9f130ed6047de1a96`
                    }
                });
                const track = response.data.tracks.items[0];
                if (track) {
                    urls[rec.name] = track.album.images[0].url;
                }
            } catch (error) {
                console.error(`Error fetching cover for ${rec.name}:`, error);
            }
        }
        setCoverUrls(urls);
    }

    const handleLike = async (song) => {
        try {
            await axios.post('http://localhost:5001/like', {
                song_name: song.name,
                song_artist: song.artists,
                song_year: song.year,
            });
            setLikedSongs([...likedSongs, song]);
        } catch (error) {
            console.error('Error liking song:', error);
        }
    };

    const goToPlaylist = () => {
        navigate('/playlist');
    };

    return (
        <div
            className="min-h-screen relative w-full bg-cover bg-center flex flex-col items-center justify-center"
            style={{ backgroundImage: `url(${JumbotronImage})` }}
        >
            <h1 className="text-white relative z-10 font-black italic text-7xl mb-10">
                SoundWave
            </h1>
            <form onSubmit={handleSubmit} className="relative z-10 bg-white p-10 rounded-lg shadow-lg">
                {songs.map((song, index) => (
                    <div key={index} className="mb-4">
                        <input
                            type="text"
                            name="name"
                            placeholder="Song Name"
                            value={song.name}
                            onChange={(event) => handleSongChange(index, event)}
                            className="p-2 border border-gray-300 rounded mr-2"
                        />
                        <input
                            type="text"
                            name="year"
                            placeholder="Year"
                            value={song.year}
                            onChange={(event) => handleSongChange(index, event)}
                            className="p-2 border border-gray-300 rounded"
                        />
                    </div>
                ))}
                <button type="button" onClick={addSong} className="p-2 bg-blue-500 text-white rounded mb-4">Add Song</button>
                <div className="mb-4">
                    <input
                        type="text"
                        placeholder="Genre"
                        value={genre}
                        onChange={(event) => setGenre(event.target.value)}
                        className="p-2 border border-gray-300 rounded w-full"
                    />
                </div>
                <div className="mb-4">
                    <input
                        type="text"
                        placeholder="Vibe"
                        value={vibe}
                        onChange={(event) => setVibe(event.target.value)}
                        className="p-2 border border-gray-300 rounded w-full"
                    />
                </div>
                <button type="submit" className="p-2 bg-green-500 text-white rounded w-full">Get Recommendations</button>
            </form>
            {loading ? (
                <div className="relative z-10 mt-10">
                    <ClipLoader color={"#ffffff"} loading={loading} size={50} />
                </div>
            ) : (
                <>
                    <h2 className="text-white relative z-10 mt-10">Recommendations</h2>
                    <ul className="relative z-10 text-white grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mt-4 mb-8 px-5 ">
                        {recommendations.map((rec, index) => (
                            <li key={index} className="bg-gray-800 p-4 rounded-lg shadow-lg flex items-center">
                                {coverUrls[rec.name] && <img src={coverUrls[rec.name]} alt={`${rec.name} cover`} width="100" className="mr-4 rounded" />}
                                <div>
                                    <h3 className="text-lg font-bold">{rec.name}</h3>
                                    <p className="text-sm">{rec.artists} ({rec.year})</p>
                                </div>
                            </li>
                        ))}
                    </ul>
                </>
            )}
            <div className="absolute inset-0 bg-gradient-to-r from-black/50 via-black/30 to-black/80"></div>
            <div className="absolute inset-0 bg-gradient-to-l from-black/70 via-transparent to-black/90"></div>
        </div>
    );
};

export default Recomendation;
