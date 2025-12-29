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

if not ROBOFLOW_API_KEY:
    raise ValueError("ROBOFLOW_API_KEY environment variable is required")
if not GITHUB_TOKEN:
    raise ValueError("GITHUB_TOKEN environment variable is required")

# ================= PERSISTENT STORAGE =================
ESP_RESULTS_FILE = "esp_results.json"
ESP_RESULTS = {}  # { esp_id: { count: int, last_update: int } }
ESP_LOCK = Lock()

# ================= GITHUB CONFIG =================
GITHUB_REPO = "onlykartika/ESP32-CAM"
GITHUB_FOLDER = "images"  # gambar tetap di folder images
GITHUB_API_IMAGES = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{GITHUB_FOLDER}"
GITHUB_API_ROOT = f"https://api.github.com/repos/{GITHUB_REPO}/contents"  # untuk esp_results.json di root
GITHUB_HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "User-Agent": "Render-AI-Server",
    "Accept": "application/vnd.github.v3+json"
}

def load_esp_results():
    global ESP_RESULTS
    # 1. Coba load dari file lokal
    if os.path.exists(ESP_RESULTS_FILE):
        try:
            with open(ESP_RESULTS_FILE, "r") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    ESP_RESULTS = data
                    print("[INFO] ESP_RESULTS loaded from local file")
                    return data
        except Exception as e:
            print(f"[ERROR] Failed loading local esp_results.json: {e}")

    # 2. Kalau gagal/kosong, fallback ke GitHub
    try:
        get_url = f"{GITHUB_API_ROOT}/esp_results.json"
        res = requests.get(get_url, headers=GITHUB_HEADERS, timeout=10)
        if res.status_code == 200:
            content_b64 = res.json()["content"]
            content = base64.b64decode(content_b64).decode('utf-8')
            data = json.loads(content)
            if isinstance(data, dict):
                ESP_RESULTS = data
                # Simpan juga ke lokal biar cepat next time
                save_esp_results()
                print("[INFO] ESP_RESULTS loaded from GitHub fallback")
                return data
        else:
            print(f"[WARN] esp_results.json not found on GitHub (status {res.status_code})")
    except Exception as e:
        print(f"[WARN] Failed loading esp_results.json from GitHub: {e}")

    # 3. Kalau semua gagal → mulai dari kosong
    ESP_RESULTS = {}
    return {}

def save_esp_results():
    try:
        with open(ESP_RESULTS_FILE, "w") as f:
            json.dump(ESP_RESULTS, f)
        print("[INFO] ESP_RESULTS saved to local file")
    except Exception as e:
        print(f"[ERROR] Failed saving local esp_results.json: {e}")

# Load saat app start
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

# ================= HEALTH CHECK =================
@app.route("/", methods=["GET"])
def health():
    return "Render AI server running - with GitHub persistent backup"

# ================= IMAGE UPLOAD =================
@app.route("/upload", methods=["POST"])
def upload():
    if not request.data:
        return jsonify({"error": "no image received"}), 400

    esp_id = request.headers.get("X-ESP-ID", "unknown")
    print(f"[INFO] Upload from ESP: {esp_id}")

    timestamp = int(time.time())
    filename = f"{esp_id}_{timestamp}.jpg"

    # Simpan gambar sementara
    try:
        with open(filename, "wb") as f:
            f.write(request.data)
    except Exception as e:
        return jsonify({"error": "failed to save image", "detail": str(e)}), 500

    # ===== ROBOFLOW INFERENCE =====
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

    # ===== UPLOAD GAMBAR KE GITHUB (best effort) =====
    try:
        with open(filename, "rb") as f:
            content = base64.b64encode(f.read()).decode()

        put_url = f"{GITHUB_API_IMAGES}/{esp_id}/{filename}"
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
            print(f"[WARN] GitHub image upload failed: {res.status_code}")
        else:
            print(f"[INFO] Image {filename} uploaded to GitHub")
    except Exception as e:
        print(f"[WARN] GitHub image upload error: {e}")

    # Hapus file sementara
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

    # ===== UPDATE & SAVE RESULT =====
    with ESP_LOCK:
        ESP_RESULTS[esp_id] = {
            "count": detected_count,
            "last_update": int(time.time() * 1000)
        }
        save_esp_results()  # simpan ke lokal

        # ===== UPLOAD ESP_RESULTS.JSON KE GITHUB (persistent backup) =====
        try:
            json_filename = "esp_results.json"
            json_content = json.dumps(ESP_RESULTS).encode('utf-8')
            content_b64 = base64.b64encode(json_content).decode('utf-8')

            # Ambil sha kalau file sudah ada
            get_url = f"{GITHUB_API_ROOT}/{json_filename}"
            get_res = requests.get(get_url, headers=GITHUB_HEADERS, timeout=10)
            sha = get_res.json().get("sha") if get_res.status_code == 200 else None

            put_url = f"{GITHUB_API_ROOT}/{json_filename}"
            put_data = {
                "message": f"Update results from {esp_id} upload at {time.strftime('%Y-%m-%d %H:%M:%S')}",
                "content": content_b64
            }
            if sha:
                put_data["sha"] = sha

            put_res = requests.put(put_url, headers=GITHUB_HEADERS, json=put_data, timeout=15)
            if put_res.status_code in (200, 201):
                print("[INFO] esp_results.json successfully uploaded to GitHub")
            else:
                print(f"[WARN] GitHub JSON upload failed: {put_res.status_code} {put_res.text}")
        except Exception as e:
            print(f"[WARN] GitHub JSON upload error: {e}")

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