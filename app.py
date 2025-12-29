import os
import time
import base64
import json
import requests
from flask import Flask, request, jsonify
from inference_sdk import InferenceHTTPClient
from threading import Lock

# ================= FLASK APP =================
app = Flask(__name__)

# ================= ENV =================
ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")

# ================= PERSISTENT STORAGE =================
ESP_RESULTS_FILE = "esp_results.json"
ESP_RESULTS = {}  # { esp_id: { count: int, last_update: int } }
ESP_LOCK = Lock()

def load_esp_results():
    if os.path.exists(ESP_RESULTS_FILE):
        try:
            with open(ESP_RESULTS_FILE, "r") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    print("[INFO] ESP_RESULTS loaded from file")
                    return data
        except Exception as e:
            print(f"[ERROR] Failed loading esp_results.json: {e}")
    return {}

def save_esp_results():
    try:
        with open(ESP_RESULTS_FILE, "w") as f:
            json.dump(ESP_RESULTS, f)
        print("[INFO] ESP_RESULTS saved")
    except Exception as e:
        print(f"[ERROR] Failed saving esp_results.json: {e}")

ESP_RESULTS = load_esp_results()

# ================= ROBOFLOW (LAZY INIT) =================
rf_client = None

def get_rf_client():
    global rf_client
    if rf_client is None:
        rf_client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key=ROBOFLOW_API_KEY
        )
    return rf_client

WORKSPACE_NAME = "my-workspace-grzes"
WORKFLOW_ID = "detect-count-and-visualize"

# ================= GITHUB =================
GITHUB_REPO = "onlykartika/ESP32-CAM"
GITHUB_FOLDER = "images"
GITHUB_API = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{GITHUB_FOLDER}"
GITHUB_HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "User-Agent": "Render-AI-Server",
    "Accept": "application/vnd.github.v3+json"
}

# ================= HEALTH CHECK =================
@app.route("/", methods=["GET"])
def health():
    return "Render AI server running"

# ================= IMAGE UPLOAD =================
@app.route("/upload", methods=["POST"])
def upload():
    if not request.data:
        return jsonify({"error": "no image received"}), 400

    esp_id = request.headers.get("X-ESP-ID", "unknown")
    print(f"[INFO] Upload from ESP: {esp_id}")

    timestamp = int(time.time())
    filename = f"{esp_id}_{timestamp}.jpg"

    try:
        with open(filename, "wb") as f:
            f.write(request.data)
    except Exception as e:
        return jsonify({"error": "failed to save image", "detail": str(e)}), 500

    # ===== ROBOFLOW =====
    try:
        rf = get_rf_client()
        result = rf.run_workflow(
            workspace_name=WORKSPACE_NAME,
            workflow_id=WORKFLOW_ID,
            images={"image": filename},
            use_cache=False
        )
    except Exception as e:
        os.remove(filename)
        return jsonify({"error": "roboflow failed", "detail": str(e)}), 500

    # ===== UPLOAD TO GITHUB (BEST EFFORT) =====
    try:
        with open(filename, "rb") as f:
            content = base64.b64encode(f.read()).decode()

        put_url = f"{GITHUB_API}/{esp_id}/{filename}"
        res = requests.put(
            put_url,
            headers=GITHUB_HEADERS,
            json={
                "message": f"upload from {esp_id} ({filename})",
                "content": content
            },
            timeout=15
        )

        if res.status_code not in (200, 201):
            print(f"[WARN] GitHub upload failed: {res.status_code}")
    except Exception as e:
        print(f"[WARN] GitHub upload error: {e}")

    try:
        os.remove(filename)
    except:
        pass

    # ===== PARSE ROBOFLOW RESULT =====
    predictions = []

    if isinstance(result, dict) and "predictions" in result:
        predictions = result["predictions"]
    elif isinstance(result, list):
        for item in result:
            if isinstance(item, dict) and "predictions" in item:
                predictions.extend(item["predictions"])

    TARGET_LABEL = "panulirus ornatus - juvenile"
    filtered = []

    for p in predictions:
        if not isinstance(p, dict):
            continue
        label = p.get("class") or p.get("label")
        conf = p.get("confidence") or p.get("score") or 0
        if label and label.lower() == TARGET_LABEL.lower() and conf >= 0.6:
            filtered.append({
                "label": label,
                "confidence": round(conf * 100, 2)
            })

    detected_count = len(filtered)

    # ===== SAVE RESULT =====
    with ESP_LOCK:
        ESP_RESULTS[esp_id] = {
            "count": detected_count,
            "last_update": int(time.time() * 1000)
        }
        save_esp_results()

    total_all = sum(v["count"] for v in ESP_RESULTS.values())

    return jsonify({
        "status": "ok",
        "esp_id": esp_id,
        "detected_this_esp": detected_count,
        "total_detected_all_esp": total_all,
        "per_esp": ESP_RESULTS,
        "objects": filtered
    }), 200

# ================= SUMMARY =================
@app.route("/summary", methods=["GET"])
def summary():
    with ESP_LOCK:
        return jsonify({
            "total_all_esp": sum(v["count"] for v in ESP_RESULTS.values()),
            "per_esp": ESP_RESULTS
        })

# ================= RUN =================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
