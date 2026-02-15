"""
Complete AI Voice + Fraud Detection System
GUVI Hackathon 2026
FULL STABLE WINDOWS VERSION
"""

import os
import warnings
warnings.filterwarnings("ignore")

# 🔥 FORCE FFMPEG PATH INTO ENVIRONMENT
AudioSegment.converter = "ffmpeg"

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import pandas as pd
from datetime import datetime
import librosa
import soundfile as sf
from scipy.stats import kurtosis, skew
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import joblib
import requests

# 🔐 HACKATHON API KEY
API_KEY = os.getenv("API_KEY")

# ================= SPEECH RECOGNITION =================

import speech_recognition as sr
from pydub import AudioSegment

# Use system ffmpeg (Linux compatible)
AudioSegment.converter = "ffmpeg"


# ================= APP INIT =================

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = os.path.join(os.getcwd(), "audio_uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

print("=" * 80)
print("🚀 INITIALIZING DUAL AI DETECTION SYSTEM")
print("=" * 80)

# =====================================================
# =============== AI VOICE DETECTOR ===================
# =====================================================

class AudioOnlyAIDetector:

    def extract_audio_only_features(self, audio_path):
        try:
            y, sr = librosa.load(audio_path, sr=16000, duration=30)
            if len(y) == 0:
                return None

            features = {}

            spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
            features["spectral_centroid_mean"] = np.mean(spec_cent)

            spec_flat = librosa.feature.spectral_flatness(y=y)
            features["spectral_flatness_mean"] = np.mean(spec_flat)

            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pitch_vals = []

            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_vals.append(pitch)

            if pitch_vals:
                features["pitch_std"] = np.std(pitch_vals)
                features["pitch_consistency"] = 1 / (np.std(pitch_vals) + 1)
            else:
                features["pitch_std"] = 0
                features["pitch_consistency"] = 0

            rms = librosa.feature.rms(y=y)
            features["energy_consistency"] = 1 / (np.std(rms) + 1e-6)

            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            features["mfcc_mean"] = np.mean(mfcc)

            features["kurtosis"] = kurtosis(y)
            features["skewness"] = skew(y)

            return features

        except Exception as e:
            print("❌ Audio feature extraction error:", e)
            return None

    def detect_ai_voice(self, audio_path):
        features = self.extract_audio_only_features(audio_path)
        if features is None:
            return False, 0, {}, []

        ai_score = 0
        indicators = []

        if features.get("pitch_consistency", 0) > 0.6:
            ai_score += 30
            indicators.append("Unnatural pitch consistency")

        if features.get("energy_consistency", 0) > 0.35:
            ai_score += 25
            indicators.append("Uniform energy distribution")

        if features.get("pitch_std", 100) < 40:
            ai_score += 20
            indicators.append("Low pitch variation")

        if features.get("spectral_flatness_mean", 0) > 0.25:
            ai_score += 15
            indicators.append("Flat spectral profile")

        ai_score = min(ai_score, 100)
        is_ai = ai_score >= 50

        return is_ai, ai_score, features, indicators


ai_voice_detector = AudioOnlyAIDetector()

# =====================================================
# ================= FRAUD DETECTOR ====================
# =====================================================

def load_fraud_dataset():
    return pd.DataFrame({
        "label": [
            1,1,1,1,1,1,1,1,1,1,
            0,0,0,0,0,0,0,0,0,0
        ],
        "message": [
            # Fraud messages
            "Share your OTP immediately",
            "Your bank account is blocked",
            "Send OTP to verify account",
            "Urgent action required now",
            "Lottery winner claim now",
            "Income tax refund pending click link",
            "Police case registered against you",
            "Your account will be suspended",
            "Provide Aadhaar details now",
            "Immediate payment required",

            # Legit messages
            "Meeting tomorrow at 5 PM",
            "Delivery arriving today",
            "Lunch at 2 PM",
            "Project deadline extended",
            "Call me when free",
            "Your order has been shipped",
            "Let us discuss the assignment",
            "Happy birthday wishes",
            "See you soon",
            "Thank you for your help"
        ]
    })


class FraudDetector:

    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=2000)
        self.model = LogisticRegression(max_iter=500)
        self.train()

    def train(self):
        df = load_fraud_dataset()
        X = df["message"]
        y = df["label"]

        X_vec = self.vectorizer.fit_transform(X)
        self.model.fit(X_vec, y)

    def predict(self, text):

        text = text.lower()

        # 🔥 Rule-based override (Hackathon Safe)
        fraud_keywords = [
            "otp", "account blocked", "urgent",
            "suspend", "verify", "bank",
            "aadhaar", "payment", "refund",
            "lottery", "police case"
        ]

        for word in fraud_keywords:
            if word in text:
                return 0.9

        X = self.vectorizer.transform([text])
        return float(self.model.predict_proba(X)[0][1])


fraud_detector = FraudDetector()

# =====================================================
# ================= SPEECH PROCESSOR ==================
# =====================================================

class SpeechProcessor:

    def __init__(self):
        self.recognizer = sr.Recognizer()

    def convert_to_wav(self, path):
        try:
            audio = AudioSegment.from_file(path)
            wav_path = path.rsplit(".", 1)[0] + "_converted.wav"
            audio.export(wav_path, format="wav", parameters=["-acodec", "pcm_s16le"])
            return wav_path
        except Exception as e:
            print("Conversion error:", e)
            return None

    def speech_to_text(self, path):
        try:
            if not path.endswith(".wav"):
                path = self.convert_to_wav(path)

            if path is None or not os.path.exists(path):
                return "Conversion failed", "Error"

            with sr.AudioFile(path) as source:
                audio_data = self.recognizer.record(source, duration=30)
                text = self.recognizer.recognize_google(audio_data, language="en-IN")
                return text, "English"

        except Exception as e:
            return f"Error: {e}", "Error"

speech_processor = SpeechProcessor()

# =====================================================
# ================= ROUTES =============================
# =====================================================

@app.route("/")
def home():
    return render_template("index.html")

# 🔥 HACKATHON ENDPOINT (JSON + AUTH)
@app.route("/api/analyze", methods=["POST"])
def analyze():
    try:
        auth_header = request.headers.get("Authorization")
        if not auth_header or auth_header != f"Bearer {API_KEY}":
            return jsonify({"status": "error", "message": "Unauthorized"}), 401

        data = request.get_json()
        if not data or "audio_url" not in data:
            return jsonify({"status": "error", "message": "audio_url is required"}), 400

        audio_url = data["audio_url"]
        response = requests.get(audio_url)

        if response.status_code != 200:
            return jsonify({"status": "error", "message": "Failed to download audio"}), 400

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(UPLOAD_FOLDER, f"call_{timestamp}.mp3")

        with open(filepath, "wb") as f:
            f.write(response.content)

        is_ai, ai_score, features, indicators = ai_voice_detector.detect_ai_voice(filepath)
        transcript, language = speech_processor.speech_to_text(filepath)

        fraud_score = 0
        if (
                transcript
                and language != "Error"
                and transcript.lower() not in ["conversion failed", "error"]
                and len(transcript.strip()) > 5
            ):

            fraud_score = fraud_detector.predict(transcript) * 100

        final_score = int(ai_score * 0.6 + fraud_score * 0.4)

        try:
            if os.path.exists(filepath):
                os.remove(filepath)
            converted_path = filepath.rsplit(".", 1)[0] + "_converted.wav"
            if os.path.exists(converted_path):
                os.remove(converted_path)
        except:
            pass

        return jsonify({
            "status": "success",
            "analysis": {
                "overall_threat_score": final_score,
                "ai_voice_score": int(ai_score),
                "fraud_content_score": int(fraud_score)
            },
            "ai_voice_detection": {
                "is_ai_generated": is_ai,
                "confidence": f"{ai_score}%",
                "indicators": indicators
            },
            "fraud_detection": {
                "transcript": transcript,
                "language": language
            }
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

# =====================================================
# ================= RUN SERVER ========================
# =====================================================

if __name__ == "__main__":
    print("🌐 Server running at http://localhost:5001\n")
    app.run(host="0.0.0.0", port=5001, debug=False)
