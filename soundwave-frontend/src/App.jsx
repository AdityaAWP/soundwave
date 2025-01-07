import React, { useState } from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import LandingPage from './components/Landing';
import Recomendation from './components/Recomendation';

const App = () => {
  const [username, setUsername] = useState('');

  return (
    <Router>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/recommendation" element={<Recomendation username={username} />} />
      </Routes>
    </Router>
  );
}

export default App;
