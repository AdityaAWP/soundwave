import React from "react";
import { Link, useNavigate } from 'react-router-dom';
import JumbotronImage from "../assets/jumbotron.jpg";

const LandingPage = () => {
    const navigate = useNavigate();


    return (
        <div
            className="min-h-screen relative w-full bg-cover bg-center flex flex-col items-center justify-center"
            style={{ backgroundImage: `url(${JumbotronImage})` }}
        >
            <h1 className="text-white relative z-10 font-black italic text-7xl mb-4">
                SoundWave
            </h1>
            <p className="text-white relative z-10 text-xl mb-8">
                Discover and enjoy your favorite music.
            </p>
            <Link to="/recommendation">
                <button type="button" className="p-2 bg-green-500 text-white rounded relative z-10">
                    Get Started
                </button>
            </Link>
            <div className="absolute inset-0 bg-gradient-to-r from-black/50 via-black/30 to-black/80 z-0"></div>
            <div className="absolute inset-0 bg-gradient-to-l from-black/70 via-transparent to-black/90 z-0"></div>
        </div>
    );
};

export default LandingPage;
