import os
import time
import base64
import requests
from flask import Flask, request, jsonify
from inference_sdk import InferenceHTTPClient
from threading import Lock

# ================= FLASK APP =================
app = Flask(__name__)

# ================= ENV =================
ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")

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
    "User-Agent": "Render-AI-Server"
}

# ================= GLOBAL STORAGE =================
ESP_RESULTS = {}          # { esp_id: detected_count }
ESP_LOCK = Lock()

# ================= HEALTH CHECK =================
@app.route("/", methods=["GET"])
def health():
    return "Render AI server running"

# ================= IMAGE UPLOAD =================
@app.route("/upload", methods=["POST"])
def upload():
    if not request.data:
        return jsonify({"error": "no image received"}), 400

    # ===== ESP ID =====
    esp_id = request.headers.get("X-ESP-ID", "unknown")

    # ===== SIMPAN IMAGE =====
    timestamp = int(time.time())
    filename = f"{esp_id}_{timestamp}.jpg"

    with open(filename, "wb") as f:
        f.write(request.data)

    # ===== RUN ROBOFLOW (AMAN UNTUK RENDER) =====
    try:
        rf = get_rf_client()
        result = rf.run_workflow(
            workspace_name=WORKSPACE_NAME,
            workflow_id=WORKFLOW_ID,
            images={"image": filename},
            use_cache=False
        )
    except Exception as e:
        return jsonify({"error": "roboflow failed", "detail": str(e)}), 500

    # ===== UPLOAD KE GITHUB =====
    try:
        with open(filename, "rb") as f:
            content = base64.b64encode(f.read()).decode()

        requests.put(
            f"{GITHUB_API}/{esp_id}/{filename}",
            headers=GITHUB_HEADERS,
            json={
                "message": f"upload from {esp_id} ({filename})",
                "content": content
            },
            timeout=15
        )
    except Exception as e:
        return jsonify({"error": "github upload failed", "detail": str(e)}), 500

    # ===== CLEANUP =====
    try:
        os.remove(filename)
    except:
        pass

    # ===== PARSE HASIL =====
    predictions = []

    if isinstance(result, dict) and "predictions" in result:
        predictions = result["predictions"]
    elif isinstance(result, list):
        for item in result:
            if isinstance(item, dict) and "predictions" in item:
                predictions.extend(item["predictions"])

    # ===== FILTER TARGET =====
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

    # ===== SIMPAN HASIL PER ESP =====
    with ESP_LOCK:
        ESP_RESULTS[esp_id] = detected_count
        total_all_esp = sum(ESP_RESULTS.values())

    # ===== RESPONSE =====
    return jsonify({
        "status": "ok",
        "esp_id": esp_id,
        "filename": filename,
        "detected_this_esp": detected_count,
        "total_detected_all_esp": total_all_esp,
        "per_esp": ESP_RESULTS,
        "objects": filtered
    }), 200


# ================= SUMMARY =================
@app.route("/summary", methods=["GET"])
def summary():
    with ESP_LOCK:
        return jsonify({
            "total_all_esp": sum(ESP_RESULTS.values()),
            "per_esp": ESP_RESULTS
        })


# ================= LOCAL RUN =================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
