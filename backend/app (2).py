from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import joblib

app = Flask(__name__)
CORS(app)

# Load models
model_dir = os.path.join(os.path.dirname(__file__), "models")
pitcher_k_model = joblib.load(os.path.join(model_dir, "pitcher_k_model.pkl"))
pitcher_ip_model = joblib.load(os.path.join(model_dir, "pitcher_ip_model.pkl"))
hitter_hr_model = joblib.load(os.path.join(model_dir, "hitter_hr_model.pkl"))
hitter_hits_model = joblib.load(os.path.join(model_dir, "hitter_hits_model.pkl"))
hitter_rbi_model = joblib.load(os.path.join(model_dir, "hitter_rbi_model.pkl"))

@app.route("/")
def health_check():
    return jsonify({"status": "API is running"})

@app.route("/predict/player-props", methods=["POST"])
def predict_player_props():
    data = request.get_json()

    pitcher_stats = data.get("pitcherStats", {})
    hitter_stats = data.get("hitterStats", {})

    def default(stats, keys):
        return [stats.get(k, 0) for k in keys]

    pitcher_input = default(pitcher_stats, ["era", "whip", "k9", "ip_per_game"])
    hitter_input = default(hitter_stats, ["avg", "obp", "slg", "hr", "rbi", "hits"])

    return jsonify({
        "pitcher": {
            "predicted_strikeouts": round(pitcher_k_model.predict([pitcher_input])[0], 2),
            "predicted_innings": round(pitcher_ip_model.predict([pitcher_input])[0], 2),
            "confidence": 0.85
        },
        "hitter": {
            "predicted_home_runs": round(hitter_hr_model.predict([hitter_input])[0], 2),
            "predicted_hits": round(hitter_hits_model.predict([hitter_input])[0], 2),
            "predicted_rbi": round(hitter_rbi_model.predict([hitter_input])[0], 2),
            "confidence": 0.81
        }
    })

@app.route("/predict/parlay", methods=["POST"])
def predict_parlay():
    threshold = request.get_json().get("confidence", 8)

    parlay = []
    picks = [
        {
            "type": "pitcher",
            "game": "Sample Game A",
            "prediction": "Strikeouts: 7.0",
            "confidence": 9
        },
        {
            "type": "hitter",
            "game": "Sample Game B",
            "prediction": "Home Runs: 1.0",
            "confidence": 8.6
        },
        {
            "type": "hitter",
            "game": "Sample Game C",
            "prediction": "Hits: 2.0",
            "confidence": 8.3
        }
    ]

    for pick in picks:
        if pick["confidence"] >= threshold:
            parlay.append(pick)
    return jsonify({"parlay": parlay[:3]})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)