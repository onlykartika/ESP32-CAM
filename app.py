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
ESP_RESULTS = {}  # { esp_id: detected_count }
ESP_LOCK = Lock()

def load_esp_results():
    """Load ESP_RESULTS dari file JSON kalau ada"""
    if os.path.exists(ESP_RESULTS_FILE):
        try:
            with open(ESP_RESULTS_FILE, "r") as f:
                data = json.load(f)
                print(f"[INFO] Loaded ESP_RESULTS from file: {data}")
                return data
        except Exception as e:
            print(f"[ERROR] Gagal load esp_results.json: {e}")
    return {}

def save_esp_results():
    """Simpan ESP_RESULTS ke file JSON"""
    try:
        with open(ESP_RESULTS_FILE, "w") as f:
            json.dump(ESP_RESULTS, f)
        print(f"[INFO] Saved ESP_RESULTS to file: {ESP_RESULTS}")
    except Exception as e:
        print(f"[ERROR] Gagal simpan esp_results.json: {e}")

# Load saat app mulai
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

    # ===== ESP ID =====
    esp_id = request.headers.get("X-ESP-ID", "unknown")
    print(f"[INFO] Upload dari ESP ID: {esp_id}")

    # ===== SIMPAN IMAGE SEMENTARA =====
    timestamp = int(time.time())
    filename = f"{esp_id}_{timestamp}.jpg"
    try:
        with open(filename, "wb") as f:
            f.write(request.data)
        print(f"[INFO] Gambar disimpan sementara: {filename}")
    except Exception as e:
        return jsonify({"error": "gagal simpan gambar", "detail": str(e)}), 500

    # ===== RUN ROBOFLOW =====
    try:
        rf = get_rf_client()
        result = rf.run_workflow(
            workspace_name=WORKSPACE_NAME,
            workflow_id=WORKFLOW_ID,
            images={"image": filename},
            use_cache=False
        )
        print("[INFO] Roboflow inference selesai")
    except Exception as e:
        os.remove(filename)  # cleanup kalau gagal
        return jsonify({"error": "roboflow failed", "detail": str(e)}), 500

    # ===== UPLOAD KE GITHUB (opsional, tetap coba) =====
    try:
        with open(filename, "rb") as f:
            content = base64.b64encode(f.read()).decode()
        put_url = f"{GITHUB_API}/{esp_id}/{filename}"
        response = requests.put(
            put_url,
            headers=GITHUB_HEADERS,
            json={
                "message": f"upload from {esp_id} ({filename})",
                "content": content
            },
            timeout=15
        )
        if response.status_code in [200, 201]:
            print("[INFO] Upload ke GitHub sukses")
        else:
            print(f"[WARNING] Upload GitHub gagal: {response.status_code} {response.text}")
    except Exception as e:
        print(f"[WARNING] Upload GitHub error: {str(e)}")

    # ===== CLEANUP FILE SEMENTARA =====
    try:
        os.remove(filename)
        print("[INFO] File sementara dihapus")
    except Exception as e:
        print(f"[WARNING] Gagal hapus file sementara: {e}")

    # ===== PARSE HASIL ROBOFLOW =====
    predictions = []
    if isinstance(result, dict) and "predictions" in result:
        predictions = result["predictions"]
    elif isinstance(result, list):
        for item in result:
            if isinstance(item, dict) and "predictions" in item:
                predictions.extend(item["predictions"])

    # ===== FILTER TARGET (juvenile lobster) =====
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

    # ===== SIMPAN HASIL KE GLOBAL + FILE =====
    with ESP_LOCK:
        ESP_RESULTS[esp_id] = detected_count
        total_all_esp = sum(ESP_RESULTS.values())
        save_esp_results()  # <--- Ini yang penting!

    # ===== RESPONSE KE ESP =====
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