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
ROBOFLOW_API_KEY_ANAKAN = os.environ.get("ROBOFLOW_API_KEY_ANAKAN")
ROBOFLOW_API_KEY_INDUKAN = os.environ.get("ROBOFLOW_API_KEY_INDUKAN")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")

if not ROBOFLOW_API_KEY_ANAKAN:
    raise ValueError("ROBOFLOW_API_KEY_ANAKAN is required")
if not ROBOFLOW_API_KEY_INDUKAN:
    raise ValueError("ROBOFLOW_API_KEY_INDUKAN is required")
if not GITHUB_TOKEN:
    raise ValueError("GITHUB_TOKEN is required")

# ================= PERSISTENT STORAGE =================
ESP_RESULTS_FILE = "esp_results.json"
ESP_RESULTS = {}
ESP_LOCK = Lock()

# ================= GITHUB CONFIG =================
GITHUB_REPO = "onlykartika/ESP32-CAM"
GITHUB_FOLDER = "images"
GITHUB_API_IMAGES = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{GITHUB_FOLDER}"
GITHUB_API_ROOT = f"https://api.github.com/repos/{GITHUB_REPO}/contents"
GITHUB_HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "User-Agent": "Render-AI-Server",
    "Accept": "application/vnd.github.v3+json"
}

# ================= LOAD / SAVE =================
def load_esp_results():
    global ESP_RESULTS
    if os.path.exists(ESP_RESULTS_FILE):
        try:
            with open(ESP_RESULTS_FILE, "r") as f:
                ESP_RESULTS = json.load(f)
                return ESP_RESULTS
        except:
            pass
    ESP_RESULTS = {}
    return ESP_RESULTS

def save_esp_results():
    with open(ESP_RESULTS_FILE, "w") as f:
        json.dump(ESP_RESULTS, f)

ESP_RESULTS = load_esp_results()

# ================= ROBOFLOW CLIENT =================
rf_anakan = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=ROBOFLOW_API_KEY_ANAKAN
)

rf_indukan = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=ROBOFLOW_API_KEY_INDUKAN
)

# ================= KOLAM MAPPING =================
KOLAM_CONFIG = {
    "anakan": {
        "esp_ids": ["esp_1", "esp_2", "esp_3"],
        "client": rf_anakan,
        "workspace": "my-workspace-grzes",
        "workflow": "detect-count-and-visualize",
        "label": "panulirus ornatus - juvenile"
    },
    "indukan": {
        "esp_ids": ["esp_4", "esp_5"],
        "client": rf_indukan,
        "workspace": "my-workspace-grzes",
        "workflow": "detect-count-and-visualize",
        "label": "panulirus ornatus - adult"
    }
}

def get_kolam(esp_id):
    for name, cfg in KOLAM_CONFIG.items():
        if esp_id in cfg["esp_ids"]:
            return name, cfg
    return None, None

# ================= HEALTH =================
@app.route("/", methods=["GET"])
def health():
    return "AI Server running (ANAKAN + INDUKAN)"

# ================= UPLOAD =================
@app.route("/upload", methods=["POST"])
def upload():
    if not request.data:
        return jsonify({"error": "no image"}), 400

    esp_id = request.headers.get("X-ESP-ID", "").lower()
    kolam_name, cfg = get_kolam(esp_id)

    if not cfg:
        return jsonify({"error": "ESP not registered"}), 400

    ts = int(time.time())
    filename = f"{esp_id}_{ts}.jpg"

    # SAVE IMAGE
    with open(filename, "wb") as f:
        f.write(request.data)

    # ROBOFLOW (KODE LAMA – AMAN)
    try:
        result = cfg["client"].run_workflow(
            workspace_name=cfg["workspace"],
            workflow_id=cfg["workflow"],
            images={"image": filename},
            use_cache=False
        )
    except Exception as e:
        os.remove(filename)
        return jsonify({"error": "roboflow failed", "detail": str(e)}), 500

    # PARSE RESULT
    predictions = []
    if isinstance(result, dict) and "predictions" in result:
        predictions = result["predictions"]
    elif isinstance(result, list):
        for r in result:
            predictions.extend(r.get("predictions", []))

    filtered = []
    for p in predictions:
        label = p.get("class") or p.get("label")
        conf = p.get("confidence") or p.get("score") or 0
        if label and label.lower() == cfg["label"].lower() and conf >= 0.6:
            filtered.append({
                "label": label,
                "confidence": round(conf * 100, 2)
            })

    detected = len(filtered)

    # SAVE RESULT
    with ESP_LOCK:
        ESP_RESULTS[esp_id] = {
            "kolam": kolam_name,
            "count": detected,
            "last_update": int(time.time() * 1000)
        }
        save_esp_results()

    # UPLOAD IMAGE TO GITHUB (OPTIONAL)
    try:
        with open(filename, "rb") as f:
            content = base64.b64encode(f.read()).decode()

        requests.put(
            f"{GITHUB_API_IMAGES}/{esp_id}/{filename}",
            headers=GITHUB_HEADERS,
            json={
                "message": f"upload {esp_id}",
                "content": content
            }
        )
    except:
        pass

    os.remove(filename)

    return jsonify({
        "status": "ok",
        "esp_id": esp_id,
        "kolam": kolam_name,
        "detected": detected,
        "objects": filtered,
        "per_esp": ESP_RESULTS
    })

# ================= SUMMARY =================
@app.route("/summary", methods=["GET"])
def summary():
    return jsonify(ESP_RESULTS)

# ================= RUN =================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
