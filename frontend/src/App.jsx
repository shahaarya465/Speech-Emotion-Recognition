import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import './App.css';

// Constants
const API_URL = 'http://127.0.0.1:5000';

//================================================
// 1. Microphone Recorder Component
//================================================
const MicRecorder = ({ onFileSelect, onBack }) => {
  const [isRecording, setIsRecording] = useState(false);
  const [error, setError] = useState(null);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);

  const startRecording = async () => {
    setError(null);
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setError('Microphone access is not supported by your browser.');
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorderRef.current = new MediaRecorder(stream);
      audioChunksRef.current = [];

      mediaRecorderRef.current.ondataavailable = (event) => {
        audioChunksRef.current.push(event.data);
      };

      mediaRecorderRef.current.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
        const audioFile = new File([audioBlob], `recording-${Date.now()}.wav`, { type: 'audio/wav' });
        onFileSelect(audioFile);
      };

      mediaRecorderRef.current.start();
      setIsRecording(true);
    } catch (err) {
      console.error("Mic access error:", err);
      setError('Microphone access was denied. Please allow microphone permissions in your browser settings.');
    }
  };

  const stopRecording = () => {
    mediaRecorderRef.current.stop();
    mediaRecorderRef.current.stream.getTracks().forEach(track => track.stop());
    setIsRecording(false);
  };

  return (
    <div className="input-method-container">
      <h3>Record Audio from Microphone</h3>
      {error && <div className="error-message small">{error}</div>}
      {!isRecording ? (
        <button onClick={startRecording} className="action-button mic">Start Recording</button>
      ) : (
        <button onClick={stopRecording} className="action-button mic stop">Stop Recording</button>
      )}
      <button onClick={onBack} className="back-button">← Back</button>
    </div>
  );
};

//================================================
// 2. Drag & Drop Component
//================================================
const DragDrop = ({ onFileSelect, onBack }) => {
  const [isDragging, setIsDragging] = useState(false);

  const handleDragOver = (e) => { e.preventDefault(); setIsDragging(true); };
  const handleDragLeave = (e) => { e.preventDefault(); setIsDragging(false); };
  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    const files = e.dataTransfer.files;
    if (files && files.length > 0 && files[0].type.startsWith('audio/')) {
      onFileSelect(files[0]);
    } else {
      alert('Please drop an audio file.');
    }
  };

  return (
    <div className="input-method-container">
      <h3>Drag & Drop Audio File</h3>
      <div
        className={`drop-zone ${isDragging ? 'dragging' : ''}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <p>Drop your audio file here</p>
      </div>
      <button onClick={onBack} className="back-button">← Back</button>
    </div>
  );
};

//================================================
// 3. File Picker Component
//================================================
const FilePicker = ({ onFileSelect, onBack }) => {
  const handleFileChange = (e) => {
    if (e.target.files[0]) onFileSelect(e.target.files[0]);
  };

  return (
    <div className="input-method-container">
      <h3>Select Audio File</h3>
      <input type="file" id="file-picker" accept="audio/*" onChange={handleFileChange} style={{ display: 'none' }} />
      <label htmlFor="file-picker" className="action-button">Choose File</label>
      <button onClick={onBack} className="back-button">← Back</button>
    </div>
  );
};


//================================================
// Main App Component
//================================================
function App() {
  const [inputMode, setInputMode] = useState(null);
  const [file, setFile] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  
  // --- THEME STATE ---
  const [theme, setTheme] = useState(() => {
    const savedTheme = localStorage.getItem('theme');
    const userPrefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
    return savedTheme || (userPrefersDark ? 'dark' : 'light');
  });

  // --- EFFECT TO APPLY THEME ---
  useEffect(() => {
    document.body.className = theme;
    localStorage.setItem('theme', theme);
  }, [theme]);

  const toggleTheme = () => {
    setTheme(prevTheme => prevTheme === 'light' ? 'dark' : 'light');
  };
  
  const handleFileSelect = (selectedFile) => {
    setFile(selectedFile);
    setInputMode(null);
    setError('');
    setPrediction(null);
  };

  const resetSelection = () => {
      setFile(null);
      setPrediction(null);
      setError('');
      setInputMode(null);
  }

  const handleSubmit = async () => {
    if (!file) { setError('No file selected.'); return; }
    setIsLoading(true);
    setPrediction(null);
    setError('');

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post(`${API_URL}/predict`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setPrediction(response.data);
    } catch (err) {
      setError(err.response?.data?.error || 'An error occurred.');
    } finally {
      setIsLoading(false);
    }
  };

  const renderInputSelection = () => (
    <div className="input-selection">
      <h2>How would you like to provide audio?</h2>
      <div className="button-group">
        <button onClick={() => setInputMode('picker')} className="choice-button">📂<br/>Upload File</button>
        <button onClick={() => setInputMode('mic')} className="choice-button">🎤<br/>Use Microphone</button>
        <button onClick={() => setInputMode('drop')} className="choice-button">💧<br/>Drag & Drop</button>
      </div>
    </div>
  );

  const renderInputMethod = () => {
    const backFunction = () => setInputMode(null);
    switch (inputMode) {
      case 'picker': return <FilePicker onFileSelect={handleFileSelect} onBack={backFunction} />;
      case 'mic': return <MicRecorder onFileSelect={handleFileSelect} onBack={backFunction} />;
      case 'drop': return <DragDrop onFileSelect={handleFileSelect} onBack={backFunction} />;
      default: return renderInputSelection();
    }
  };

  return (
    <div className="container">
      <header>
        <h1>Speech Emotion Recognition</h1>
        <button onClick={toggleTheme} className="theme-toggle">
          {theme === 'light' ? '🌙' : '☀️'}
        </button>
      </header>
      
      <main>
        {!file ? (
            renderInputMethod()
        ) : (
          <div className="prediction-workflow">
            <div className="file-display">
                Selected File: <strong>{file.name}</strong>
                <button onClick={resetSelection} className="change-file-button">Change</button>
            </div>
            <button onClick={handleSubmit} className="predict-button" disabled={isLoading}>
              {isLoading ? 'Analyzing...' : 'Predict Emotion'}
            </button>
          </div>
        )}

        {error && <div className="error-message">{error}</div>}

        {prediction && (
      <div className="results-card">
        <h2>Prediction Results ✨</h2>
        <div className="result-item">
          <span className="result-label">Predicted Emotion:</span>
          <span className="result-value emotion">{prediction.predicted_emotion}</span>
        </div>
        <div className="result-item">
          <span className="result-label">Predicted Energy:</span>
          <span className="result-value">{prediction.predicted_energy}</span>
        </div>
        <div className="result-item">
          <span className="result-label">Confidence:</span>
          <span className="result-value">{prediction.confidence}</span>
        </div>
      </div>
    )}
      </main>
    </div>
  );
}

export default App;