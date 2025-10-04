import os
import warnings
import numpy as np
import librosa
import joblib
import tensorflow as tf
import scipy.stats
from flask import Flask, request, jsonify
from flask_cors import CORS
import soundfile as sf
import traceback
import subprocess # Import the subprocess module

# --- Basic Setup ---
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Initialize the Flask app
app = Flask(__name__)

origins = [
    "http://localhost:5173",
    "http://localhost:3000",
]

# --- Enable CORS ---
CORS(app, origins=origins, supports_credentials=True)

# --- Load All Models, Scalers, and Encoders ---
PIPELINE_OBJECTS = {}

def load_pipeline_objects(base_path="export/"):
    """
    Loads all the necessary .joblib and .keras files for the pipeline.
    """
    print("--- Loading all pipeline objects ---")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    absolute_base_path = os.path.join(script_dir, base_path)
    print(f"Attempting to load from: {absolute_base_path}")

    objects = {}
    files_to_load = {
        "router_model": "router_model.joblib",
        "router_scaler": "router_scaler.joblib",
        "router_encoder": "router_encoder.joblib",
        "specialist_high_model": "specialist_high.keras",
        "specialist_high_scaler": "scaler_h.joblib",
        "specialist_high_encoder": "le_high.joblib",
        "specialist_low_model": "specialist_low.keras",
        "specialist_low_scaler": "scaler_l.joblib",
        "specialist_low_encoder": "le_low.joblib",
    }

    try:
        for name, filename in files_to_load.items():
            full_path = os.path.join(absolute_base_path, filename)
            if not os.path.exists(full_path):
                raise FileNotFoundError(f"Required file not found: {full_path}")

            print(f"Loading: {full_path}")
            if filename.endswith(".keras"):
                objects[name] = tf.keras.models.load_model(full_path)
            elif filename.endswith(".joblib"):
                objects[name] = joblib.load(full_path)
        print("--- All pipeline objects loaded successfully! ---")
        return objects
    except Exception as e:
        print(f"FATAL ERROR during model loading: {e}")
        return None

PIPELINE_OBJECTS = load_pipeline_objects()

# --- Feature Extraction Function ---
def extract_features_detailed(file_path):
    """
    Extracts a comprehensive set of features from an audio file.
    """
    try:
        y, sr = librosa.load(file_path, sr=None)
        features = []
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        mfcc_features = np.concatenate([mfcc, mfcc_delta, mfcc_delta2], axis=0)
        for row in mfcc_features:
            features.extend([np.mean(row), np.std(row), scipy.stats.skew(row), scipy.stats.kurtosis(row)])
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        for row in chroma:
            features.extend([np.mean(row), np.std(row)])
        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        for row in mel_db:
            features.extend([np.mean(row), np.std(row)])
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        for row in contrast:
            features.extend([np.mean(row), np.std(row)])
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
        for row in tonnetz:
            features.extend([np.mean(row), np.std(row)])
        zcr = librosa.feature.zero_crossing_rate(y)
        features.extend([np.mean(zcr), np.std(zcr)])
        rmse = librosa.feature.rms(y=y)
        features.extend([np.mean(rmse), np.std(rmse)])
        return np.array(features)
    except Exception as e:
        print(f"Error during feature extraction: {e}")
        return None

# --- WARM-UP ---
if PIPELINE_OBJECTS:
    print("--- Running warm-up call to compile audio functions ---")
    try:
        sr_warmup = 22050
        y_warmup = np.zeros(sr_warmup, dtype=np.float32)
        warmup_file = "warmup_silent.wav"
        sf.write(warmup_file, y_warmup, sr_warmup)
        extract_features_detailed(warmup_file)
        os.remove(warmup_file)
        print("--- Warm-up complete, application is ready. ---")
    except Exception as e:
        print(f"An error occurred during warm-up: {e}")
# --- END WARM-UP ---

# --- Prediction Endpoint ---
@app.route("/predict", methods=["POST"])
def predict():
    """
    Processes an audio file through the two-stage router/specialist pipeline.
    """
    if not PIPELINE_OBJECTS:
        return jsonify({"error": "Pipeline models not loaded. Check server logs."}), 500

    if "file" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["file"]
    
    temp_input_file = "temp_audio_input.tmp"
    temp_wav_file = "temp_audio_converted.wav"
    ffmpeg_executable = "D:\\ffmpeg\\bin\\ffmpeg.exe" # Full path to ffmpeg

    try:
        audio_file.save(temp_input_file)

        # --- FINAL FIX: Manually call ffmpeg using subprocess ---
        print(f"Attempting to convert {temp_input_file} to {temp_wav_file} using FFmpeg...")
        
        # Construct the command. -y overwrites the output file if it exists.
        command = [ffmpeg_executable, "-i", temp_input_file, "-y", temp_wav_file]
        
        # Execute the command
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        
        print("FFmpeg conversion successful.")
        # --- END FIX ---
        
        # Extract features from the newly created WAV file
        features = extract_features_detailed(temp_wav_file)

    except subprocess.CalledProcessError as e:
        # This error is caught if ffmpeg returns a non-zero exit code (i.e., it failed)
        print("--- FFMPEG CONVERSION FAILED ---")
        print(f"FFmpeg stderr: {e.stderr}")
        traceback.print_exc()
        return jsonify({"error": "FFmpeg failed to convert the audio file."}), 500
    except FileNotFoundError:
        # This error is caught if the ffmpeg_executable path is still wrong
        print(f"--- FFMPEG NOT FOUND AT: {ffmpeg_executable} ---")
        traceback.print_exc()
        return jsonify({"error": "Server is misconfigured; FFmpeg executable not found."}), 500
    except Exception as e:
        # Catch any other errors
        print("--- AN UNEXPECTED ERROR OCCURRED ---")
        traceback.print_exc()
        return jsonify({"error": "An unexpected error occurred during file processing."}), 500
    
    finally:
        # Clean up temporary files
        if os.path.exists(temp_input_file):
            os.remove(temp_input_file)
        if os.path.exists(temp_wav_file):
            os.remove(temp_wav_file)

    if features is None:
        return jsonify({"error": "Could not extract features from the audio file."}), 500

    features_2d = features.reshape(1, -1)

    # --- STAGE 1: ROUTER MODEL PREDICTION ---
    router_scaler = PIPELINE_OBJECTS["router_scaler"]
    features_scaled_router = router_scaler.transform(features_2d)
    router_model = PIPELINE_OBJECTS["router_model"]
    energy_prediction_index = router_model.predict(features_scaled_router)[0]
    router_encoder = PIPELINE_OBJECTS["router_encoder"]
    predicted_energy = router_encoder.classes_[energy_prediction_index]
    print(f"Router prediction: Audio has '{predicted_energy}' energy.")

    # --- STAGE 2: SPECIALIST MODEL PREDICTION ---
    if predicted_energy == "high":
        scaler = PIPELINE_OBJECTS["specialist_high_scaler"]
        model = PIPELINE_OBJECTS["specialist_high_model"]
        encoder = PIPELINE_OBJECTS["specialist_high_encoder"]
    else:
        scaler = PIPELINE_OBJECTS["specialist_low_scaler"]
        model = PIPELINE_OBJECTS["specialist_low_model"]
        encoder = PIPELINE_OBJECTS["specialist_low_encoder"]

    features_scaled_specialist = scaler.transform(features_2d)
    features_reshaped = features_scaled_specialist.reshape((1, features_scaled_specialist.shape[1], 1))
    prediction_probabilities = model.predict(features_reshaped)[0]
    predicted_class_index = np.argmax(prediction_probabilities)
    confidence = np.max(prediction_probabilities)
    final_emotion = encoder.classes_[predicted_class_index]

    return jsonify({
        "predicted_energy": predicted_energy,
        "predicted_emotion": final_emotion,
        "confidence": f"{confidence * 100:.2f}%",
    })

# --- Health Check Endpoint ---
@app.route("/", methods=["GET"])
def health_check():
    if PIPELINE_OBJECTS:
        return f"API is running. {len(PIPELINE_OBJECTS)} pipeline objects loaded successfully."
    else:
        return "API is running, but pipeline objects failed to load.", 500

# --- Run the App ---
if __name__ == "__main__":
    app.run(port=5000, debug=True)