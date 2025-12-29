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

# ================= ROBOFLOW =================
client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=ROBOFLOW_API_KEY
)

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

# ================= GLOBAL STORAGE (BARU) =================
ESP_RESULTS = {}          # { esp_id: total_detected }
ESP_LOCK = Lock()         # aman untuk request bersamaan

# ================= HEALTH CHECK =================
@app.route("/", methods=["GET"])
def health():
    return "Render AI server running"

# ================= IMAGE UPLOAD =================
@app.route("/upload", methods=["POST"])
def upload():
    if not request.data:
        return jsonify({"error": "no image received"}), 400

    # ===== AMBIL ESP ID (BARU) =====
    esp_id = request.headers.get("X-ESP-ID", "unknown")

    # ===== SIMPAN IMAGE SEMENTARA =====
    timestamp = int(time.time())
    filename = f"{esp_id}_{timestamp}.jpg"

    with open(filename, "wb") as f:
        f.write(request.data)

    # ===== RUN ROBOFLOW WORKFLOW =====
    try:
        result = client.run_workflow(
            workspace_name=WORKSPACE_NAME,
            workflow_id=WORKFLOW_ID,
            images={"image": filename},
            use_cache=False
        )
    except Exception as e:
        return jsonify({"error": "roboflow failed", "detail": str(e)}), 500

    # ===== UPLOAD IMAGE KE GITHUB (PER ESP) =====
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

    # ===== CLEANUP FILE LOKAL =====
    try:
        os.remove(filename)
    except:
        pass

    # ===== PARSE HASIL (AMAN) =====
    predictions = []

    if isinstance(result, dict) and "predictions" in result:
        predictions = result["predictions"]

    elif isinstance(result, list):
        for item in result:
            if isinstance(item, dict) and "predictions" in item:
                predictions.extend(item["predictions"])

    # ===== FILTER OBJEK TARGET =====
    TARGET_LABEL = "panulirus ornatus - juvenile"
    filtered = []

    for p in predictions:
        if not isinstance(p, dict):
            continue

        label = p.get("class") or p.get("label")
        conf = p.get("confidence") or p.get("score") or 0

        if label and label.lower() == TARGET_LABEL.lower():
            if conf >= 0.6:  # threshold aman
                filtered.append({
                    "label": label,
                    "confidence": round(conf * 100, 2)
                })

    detected_count = len(filtered)

    # ===== SIMPAN HASIL PER ESP (BARU) =====
    with ESP_LOCK:
        ESP_RESULTS[esp_id] = detected_count

    # ===== HITUNG TOTAL SEMUA ESP (BARU) =====
    with ESP_LOCK:
        total_all_esp = sum(ESP_RESULTS.values())

    # ===== RESPONSE JSON =====
    response = {
        "status": "ok",
        "esp_id": esp_id,
        "filename": filename,
        "detected_this_esp": detected_count,
        "total_detected_all_esp": total_all_esp,
        "per_esp": ESP_RESULTS,
        "objects": filtered
    }

    return jsonify(response), 200


# ================= SUMMARY ENDPOINT (BARU) =================
@app.route("/summary", methods=["GET"])
def summary():
    with ESP_LOCK:
        return jsonify({
            "total_all_esp": sum(ESP_RESULTS.values()),
            "per_esp": ESP_RESULTS
        })


# ================= RUN LOCAL (Render pakai gunicorn) =================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
