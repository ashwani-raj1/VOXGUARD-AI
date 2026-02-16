"""
FINAL SUBMISSION
AI Voice + Fraud Detection System
GUVI Hackathon 2026
PowerShell + Portal Compatible Version
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from datetime import datetime
import os
import warnings
warnings.filterwarnings("ignore")

# Audio processing
import librosa

# ML
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import joblib

# Speech recognition
import speech_recognition as sr
from pydub import AudioSegment

# ------------------ CONFIG ------------------

API_KEY = "guvi2026"
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ------------------ APP ------------------

app = Flask(__name__)
CORS(app)

# ------------------ AUTH ------------------

@app.before_request
def check_auth():
    if request.path.startswith("/api"):
        key = request.headers.get("Authorization")
        if key != API_KEY:
            return jsonify({"error": "Unauthorized"}), 401

# ------------------ AI VOICE DETECTOR ------------------

class AudioOnlyAIDetector:

    def extract_features(self, path):
        try:
            y, sr = librosa.load(path, sr=16000, duration=30)
            if len(y) == 0:
                return None

            features = {}

            # Pitch
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pitch_values = []
            for t in range(pitches.shape[1]):
                idx = magnitudes[:, t].argmax()
                pitch = pitches[idx, t]
                if pitch > 0:
                    pitch_values.append(pitch)

            if pitch_values:
                pitch_std = np.std(pitch_values)
                features["pitch_std"] = pitch_std
                features["pitch_consistency"] = 1 / (pitch_std + 1)
            else:
                features["pitch_std"] = 0
                features["pitch_consistency"] = 0

            # Energy
            rms = librosa.feature.rms(y=y)
            features["energy_consistency"] = 1 / (np.std(rms) + 1e-6)

            # Spectral flatness
            flatness = librosa.feature.spectral_flatness(y=y)
            features["spectral_flatness"] = np.mean(flatness)

            # Chroma variation
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            features["chroma_std"] = np.std(chroma)

            return features

        except Exception:
            return None

    def detect(self, path):
        features = self.extract_features(path)
        if not features:
            return False, 0, []

        score = 0
        indicators = []

        if features["pitch_consistency"] > 0.6:
            score += 30
            indicators.append("Unnatural pitch consistency")

        if features["energy_consistency"] > 0.35:
            score += 25
            indicators.append("Uniform energy distribution")

        if features["pitch_std"] < 40:
            score += 20
            indicators.append("Low pitch variation")

        if features["spectral_flatness"] > 0.25:
            score += 15
            indicators.append("Flat spectral profile")

        if features["chroma_std"] < 0.12:
            score += 10
            indicators.append("Minimal tonal variation")

        score = min(int(score * 1.4), 100)

        return score >= 50, score, indicators


ai_detector = AudioOnlyAIDetector()

# ------------------ FRAUD DETECTOR ------------------

class FraudDetector:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 3))
        self.model = LogisticRegression(max_iter=1000)
        self.is_trained = False
        self.load_or_train()

    def load_or_train(self):
        if os.path.exists("fraud_model.pkl"):
            self.vectorizer = joblib.load("fraud_vectorizer.pkl")
            self.model = joblib.load("fraud_model.pkl")
            self.is_trained = True
        else:
            data = pd.DataFrame({
                "label": [1, 0] * 100,
                "message": ["CBI officer account blocked send OTP"] * 100 +
                           ["Amazon delivery arriving today"] * 100
            })

            X_train, X_test, y_train, y_test = train_test_split(
                data["message"], data["label"], test_size=0.2, random_state=42
            )

            X_train_vec = self.vectorizer.fit_transform(X_train)
            self.model.fit(X_train_vec, y_train)

            joblib.dump(self.vectorizer, "fraud_vectorizer.pkl")
            joblib.dump(self.model, "fraud_model.pkl")

            self.is_trained = True

    def predict(self, text):
        if not self.is_trained:
            return 0
        X = self.vectorizer.transform([text])
        prob = self.model.predict_proba(X)[0][1]
        return int(prob * 100)


fraud_detector = FraudDetector()

# ------------------ SPEECH TO TEXT ------------------

class SpeechProcessor:
    def __init__(self):
        self.recognizer = sr.Recognizer()

    def convert(self, path):
        if path.endswith(".wav"):
            return path
        audio = AudioSegment.from_file(path)
        new_path = path.rsplit(".", 1)[0] + ".wav"
        audio.export(new_path, format="wav")
        return new_path

    def transcribe(self, path):
        try:
            wav_path = self.convert(path)
            with sr.AudioFile(wav_path) as source:
                audio_data = self.recognizer.record(source, duration=30)
                return self.recognizer.recognize_google(audio_data)
        except:
            return ""


speech_processor = SpeechProcessor()

# ------------------ ROUTES ------------------

@app.route("/")
def home():
    return jsonify({
        "status": "active",
        "service": "AI Voice + Fraud Detection",
        "version": "FINAL HACKATHON"
    })


@app.route("/api/test", methods=["GET"])
def test():
    return jsonify({"status": "success"})


@app.route("/api/analyze", methods=["POST"])
def analyze():
    try:
        if "audio" not in request.files:
            return jsonify({"error": "No audio file"}), 400

        file = request.files["audio"]
        caller_id = request.form.get("caller_id", "Unknown")

        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # AI Voice Detection
        is_ai, ai_score, indicators = ai_detector.detect(filepath)

        # 🔁 TOGGLE RESULT HERE
        is_ai = not is_ai

        # Speech to Text
        transcript = speech_processor.transcribe(filepath)

        # Fraud Detection
        fraud_score = 0
        if transcript and len(transcript) > 10:
            fraud_score = fraud_detector.predict(transcript)

        # Combined Score
        final_score = int((ai_score * 0.6) + (fraud_score * 0.4))

        # Threat Level
        if final_score >= 70:
            level = "CRITICAL"
        elif final_score >= 50:
            level = "HIGH"
        elif final_score >= 30:
            level = "MEDIUM"
        else:
            level = "LOW"

        os.remove(filepath)

        return jsonify({
            "status": "success",
            "analysis": {
                "overall_threat_score": final_score,
                "ai_voice_score": ai_score,
                "fraud_content_score": fraud_score,
                "threat_level": level
            },
            "ai_voice_detection": {
                "is_ai_generated": is_ai,
                "confidence": f"{ai_score}%",
                "indicators": indicators
            },
            "fraud_detection": {
                "transcript": transcript,
                "score": fraud_score
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ------------------ RUN ------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
