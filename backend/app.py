
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Load models
model_dir = os.path.join(os.path.dirname(__file__), 'models')
pitcher_k_model = joblib.load(os.path.join(model_dir, 'pitcher_k_model.pkl'))
pitcher_ip_model = joblib.load(os.path.join(model_dir, 'pitcher_ip_model.pkl'))
hitter_hr_model = joblib.load(os.path.join(model_dir, 'hitter_hr_model.pkl'))
hitter_hits_model = joblib.load(os.path.join(model_dir, 'hitter_hits_model.pkl'))
hitter_rbi_model = joblib.load(os.path.join(model_dir, 'hitter_rbi_model.pkl'))

# Provide fallback averages
DEFAULT_PITCHER_INPUT = [5.0, 6.0, 2.0, 2.0]  # IP, H, BB, ER
DEFAULT_HITTER_INPUT = [4, 2, 1, 2]           # AB, H, BB, RBI

@app.route('/predict/player-props', methods=['POST'])
def predict_player_props():
    data = request.get_json()

    # Get pitcher stats or use default
    pstats = data.get('pitcherStats', {})
    pitcher_input = [
        float(pstats.get('IP', DEFAULT_PITCHER_INPUT[0])),
        float(pstats.get('H', DEFAULT_PITCHER_INPUT[1])),
        float(pstats.get('BB', DEFAULT_PITCHER_INPUT[2])),
        float(pstats.get('ER', DEFAULT_PITCHER_INPUT[3])),
    ]

    # Get hitter stats or use default
    hstats = data.get('hitterStats', {})
    hitter_input = [
        int(hstats.get('AB', DEFAULT_HITTER_INPUT[0])),
        int(hstats.get('H', DEFAULT_HITTER_INPUT[1])),
        int(hstats.get('BB', DEFAULT_HITTER_INPUT[2])),
        int(hstats.get('RBI', DEFAULT_HITTER_INPUT[3])),
    ]

    # Run predictions
    pitcher_k = float(pitcher_k_model.predict([pitcher_input])[0])
    pitcher_ip = float(pitcher_ip_model.predict([pitcher_input])[0])
    hitter_hr = float(hitter_hr_model.predict([hitter_input])[0])
    hitter_hits = float(hitter_hits_model.predict([hitter_input])[0])
    hitter_rbi = float(hitter_rbi_model.predict([hitter_input])[0])

    # Dummy confidence for now
    pitcher_conf = 9  # scale of 0–10
    hitter_conf = 8   # scale of 0–10

    return jsonify({
        "pitcher": {
            "predicted_strikeouts": round(pitcher_k, 1),
            "predicted_innings": round(pitcher_ip, 1),
            "confidence": pitcher_conf
        },
        "hitter": {
            "predicted_hits": round(hitter_hits, 1),
            "predicted_hr": round(hitter_hr, 1),
            "predicted_rbi": round(hitter_rbi, 1),
            "confidence": hitter_conf
        }
    })

@app.route("/", methods=["GET"])
def index():
    return "MLB Betting API with real models is running"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
