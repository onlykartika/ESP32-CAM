import os
import time
import json
from flask import Flask, request, jsonify
from inference_sdk import InferenceHTTPClient
from threading import Lock

# ================= FLASK =================
app = Flask(__name__)

# ================= ENV =================
API_KEY_INDUKAN = os.environ.get("ROBOFLOW_API_KEY_INDUKAN")
API_KEY_ANAKAN = os.environ.get("ROBOFLOW_API_KEY_ANAKAN")

if not API_KEY_INDUKAN or not API_KEY_ANAKAN:
    raise ValueError("Roboflow API keys required")

# ================= ROBOFLOW CLIENT =================
rf_indukan = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=API_KEY_INDUKAN
)

rf_anakan = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=API_KEY_ANAKAN
)

# ================= STORAGE =================
ESP_RESULTS_FILE = "esp_results.json"
ESP_RESULTS = {}
LOCK = Lock()

if os.path.exists(ESP_RESULTS_FILE):
    with open(ESP_RESULTS_FILE, "r") as f:
        ESP_RESULTS = json.load(f)

# ================= KONFIGURASI KOLAM =================
KOLAM = {
    "anakan": {
        "esp_ids": ["esp_1", "esp_2", "esp_3", "esp_4"],
        "client": rf_anakan,
        "workspace": "my-workspace-grzes",
        "workflow": "detect-count-and-visualize",
        "label": "panulirus ornatus - juvenile",
        "conf": 0.6
    },
    "indukan": {
        "esp_ids": ["esp_5", "esp_6", "esp_7", "esp_8"],
        "client": rf_indukan,
        "workspace": "my-workspace-rrwxa",
        "workflow": "detect-count-and-visualize",
        "label": "female",
        "conf": 0.6
    }
}

# ================= ROUTES =================
@app.route("/", methods=["GET"])
def health():
    return "AI Server OK — 2 Roboflow API, JSON Active"

@app.route("/upload", methods=["POST"])
def upload():
    if not request.data:
        return jsonify({"error": "no image"}), 400

    esp_id = request.headers.get("X-ESP-ID", "").lower()
    print(f"[INFO] ESP upload: {esp_id}")

    # Tentukan kolam
    kolam_name = None
    for k, v in KOLAM.items():
        if esp_id in v["esp_ids"]:
            kolam_name = k
            break

    if not kolam_name:
        return jsonify({"error": "unknown esp id"}), 400

    cfg = KOLAM[kolam_name]

    # Simpan gambar
    filename = f"{esp_id}_{int(time.time())}.jpg"
    with open(filename, "wb") as f:
        f.write(request.data)

    # ===== ROBOFLOW INFERENCE (FILE PATH — AMAN) =====
    try:
        result = cfg["client"].run_workflow(
            workspace_name=cfg["workspace"],
            workflow_id=cfg["workflow"],
            images={"image": filename},
            use_cache=False
        )
    except Exception as e:
        os.remove(filename)
        print("[ERROR] Roboflow:", e)
        return jsonify({"error": "roboflow failed"}), 500

    os.remove(filename)

    # ===== PARSE RESULT =====
    predictions = []
    if isinstance(result, list):
        for r in result:
            predictions.extend(r.get("predictions", []))
    elif isinstance(result, dict):
        predictions = result.get("predictions", [])

    detected = [
        p for p in predictions
        if (p.get("class") or p.get("label")) == cfg["label"]
        and (p.get("confidence", 0) >= cfg["conf"])
    ]

    count = len(detected)

    # ===== SAVE JSON =====
    with LOCK:
        ESP_RESULTS[esp_id] = {
            "kolam": kolam_name,
            "count": count,
            "last_update": int(time.time() * 1000)
        }
        with open(ESP_RESULTS_FILE, "w") as f:
            json.dump(ESP_RESULTS, f, indent=2)

    return jsonify({
        "status": "ok",
        "esp_id": esp_id,
        "kolam": kolam_name,
        "detected": count,
        "all": ESP_RESULTS
    })

@app.route("/summary", methods=["GET"])
def summary():
    return jsonify(ESP_RESULTS)

# ================= RUN =================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
