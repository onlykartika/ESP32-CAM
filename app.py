import os
import time
import json
from flask import Flask, request, jsonify
from inference_sdk import InferenceHTTPClient

app = Flask(__name__)

ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY_ANAKAN")
if not ROBOFLOW_API_KEY:
    raise ValueError("ROBOFLOW_API_KEY_ANAKAN not set")

rf = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=ROBOFLOW_API_KEY
)

MODEL_ID_ANAKAN = "panulirus-ornatus-juvenile/1"

RESULT_FILE = "esp_results.json"
if not os.path.exists(RESULT_FILE):
    with open(RESULT_FILE, "w") as f:
        json.dump({}, f)

@app.route("/", methods=["GET"])
def health():
    return "OK"

@app.route("/upload", methods=["POST"])
def upload():
    try:
        raw = request.get_data()
        if not raw or len(raw) < 1000:
            return jsonify({"error": "image empty"}), 400

        esp_id = request.headers.get("X-ESP-ID", "unknown")
        ts = int(time.time())
        filename = f"/tmp/{esp_id}_{ts}.jpg"

        with open(filename, "wb") as f:
            f.write(raw)

        # 🔥 INFERENCE (TANPA confidence)
        result = rf.infer(
            filename,
            model_id=MODEL_ID_ANAKAN
        )

        detections = result.get("predictions", [])
        count = len(detections)

        with open(RESULT_FILE, "r") as f:
            data = json.load(f)

        data[esp_id] = {
            "count": count,
            "last_update": int(time.time() * 1000)
        }

        with open(RESULT_FILE, "w") as f:
            json.dump(data, f)

        return jsonify({
            "status": "ok",
            "esp_id": esp_id,
            "count": count
        })

    except Exception as e:
        print("SERVER ERROR:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
