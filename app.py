import os
import time
import base64
import json
import requests
from flask import Flask, request, jsonify
from inference_sdk import InferenceHTTPClient
from threading import Lock

app = Flask(__name__)

# ================= ENV (DUA API KEY!) =================
ROBOFLOW_API_KEY_INDUKAN = os.environ.get("ROBOFLOW_API_KEY_INDUKAN")
ROBOFLOW_API_KEY_ANAKAN = os.environ.get("ROBOFLOW_API_KEY_ANAKAN")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")

if not ROBOFLOW_API_KEY_INDUKAN:
    raise ValueError("ROBOFLOW_API_KEY_INDUKAN required")
if not ROBOFLOW_API_KEY_ANAKAN:
    raise ValueError("ROBOFLOW_API_KEY_ANAKAN required")
if not GITHUB_TOKEN:
    raise ValueError("GITHUB_TOKEN required")

# ================= STORAGE =================
ESP_RESULTS_FILE = "esp_results.json"
ESP_RESULTS = {}
ESP_LOCK = Lock()

# ================= GITHUB (SATU REPO) =================
GITHUB_REPO = "onlykartika/ESP32-CAM"
GITHUB_FOLDER = "images"
GITHUB_API_ROOT = f"https://api.github.com/repos/{GITHUB_REPO}/contents"
GITHUB_HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "User-Agent": "Render-AI-Server",
    "Accept": "application/vnd.github.v3+json"
}

# ================= LOAD/SAVE =================
def load_esp_results():
    global ESP_RESULTS
    if os.path.exists(ESP_RESULTS_FILE):
        try:
            with open(ESP_RESULTS_FILE, "r") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    ESP_RESULTS = data
                    print("[INFO] Loaded results from local")
        except Exception as e:
            print(f"[ERROR] Load local failed: {e}")

    try:
        res = requests.get(f"{GITHUB_API_ROOT}/esp_results.json", headers=GITHUB_HEADERS, timeout=10)
        if res.status_code == 200:
            content = base64.b64decode(res.json()["content"]).decode("utf-8")
            ESP_RESULTS = json.loads(content)
            save_esp_results()
            print("[INFO] Loaded results from GitHub")
    except Exception as e:
        print(f"[WARN] GitHub load failed: {e}")

    if not isinstance(ESP_RESULTS, dict):
        ESP_RESULTS = {}

def save_esp_results():
    try:
        with open(ESP_RESULTS_FILE, "w") as f:
            json.dump(ESP_RESULTS, f, indent=2)
        print("[INFO] Saved locally")
    except Exception as e:
        print(f"[ERROR] Save local failed: {e}")

load_esp_results()

# ================= ROBOFLOW CLIENTS (DUA CLIENT TERPISAH) =================
rf_client_indukan = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",  # atau https://detect.roboflow.com kalau pakai yang lama
    api_key=ROBOFLOW_API_KEY_INDUKAN
)

rf_client_anakan = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=ROBOFLOW_API_KEY_ANAKAN
)

# ================= KONFIGURASI KOLAM =================
KOLAM_CONFIG = {
    "indukan": {
        "esp_ids": ["esp_5", "esp_6", "esp_7", "esp_8"],
        "client": rf_client_indukan,
        "workspace": "my-workspace-rrwxa",                  # Ganti kalau perlu
        "workflow": "detect-count-and-visualize",           # atau workflow ID kamu
        "target_label": "female",
        "conf_threshold": 0.6,
        "folder": "kolam_indukan"
    },
    "anakan": {
        "esp_ids": ["esp_1", "esp_2", "esp_3", "esp_4"],
        "client": rf_client_anakan,
        "workspace": "my-workspace-grzes",
        "workflow": "detect-count-and-visualize",
        "target_label": "panulirus ornatus - juvenile",
        "conf_threshold": 0.6,
        "folder": "kolam_anakan"
    }
}

# ================= ROUTES =================
@app.route("/", methods=["GET"])
def health():
    return "AI Server LOVIA - DUA API KEY (Indukan & Anakan) - Running!"

@app.route("/upload", methods=["POST"])
def upload():
    if not request.data:
        return jsonify({"error": "no image"}), 400

    esp_id = request.headers.get("X-ESP-ID", "unknown").lower()
    print(f"[INFO] Upload dari ESP: {esp_id}")

    # Tentukan kolam
    if esp_id in KOLAM_CONFIG["indukan"]["esp_ids"]:
        kolam = "indukan"
    elif esp_id in KOLAM_CONFIG["anakan"]["esp_ids"]:
        kolam = "anakan"
    else:
        return jsonify({"error": f"ESP_ID {esp_id} tidak dikenali"}), 400

    config = KOLAM_CONFIG[kolam]
    client = config["client"]
    print(f"[INFO] {esp_id} → Kolam {kolam.upper()} (workspace: {config['workspace']})")

    timestamp = int(time.time())
    filename = f"{esp_id}_{timestamp}.jpg"

    # Simpan sementara
    try:
        with open(filename, "wb") as f:
            f.write(request.data)
    except Exception as e:
        return jsonify({"error": "save failed"}), 500

    # Inference dengan client yang sesuai
    try:
        result = client.run_workflow(
            workspace_name=config["workspace"],
            workflow_id=config["workflow"],
            images={"image": open(filename, "rb")},
            use_cache=False
        )
    except Exception as e:
        os.remove(filename)
        print(f"[ERROR] Roboflow inference ({kolam}): {e}")
        return jsonify({"error": "roboflow failed", "detail": str(e)}), 500

    # Upload gambar ke GitHub (opsional)
    try:
        with open(filename, "rb") as f:
            content_b64 = base64.b64encode(f.read()).decode()
        file_path = f"{GITHUB_FOLDER}/{config['folder']}/{esp_id}/{filename}"
        put_url = f"{GITHUB_API_ROOT}/{file_path}"
        res = requests.put(put_url, headers=GITHUB_HEADERS, json={
            "message": f"Upload {esp_id} - {kolam}",
            "content": content_b64,
            "branch": "master"
        }, timeout=15)
        if res.status_code in (200, 201):
            print(f"[INFO] Gambar uploaded: {file_path}")
    except Exception as e:
        print(f"[WARN] Upload gambar gagal: {e}")

    if os.path.exists(filename):
        os.remove(filename)

    # Parse hasil deteksi
    predictions = []
    if isinstance(result, dict) and "predictions" in result:
        predictions = result.get("predictions", [])
    elif isinstance(result, list):
        for item in result:
            if isinstance(item, dict) and "predictions" in item:
                predictions.extend(item["predictions"])

    filtered = []
    target_label_lower = config["target_label"].lower()
    for p in predictions:
        if not isinstance(p, dict):
            continue
        label = p.get("class") or p.get("label") or ""
        conf = p.get("confidence") or p.get("score") or 0
        if label.lower() == target_label_lower and conf >= config["conf_threshold"]:
            filtered.append({"label": label, "confidence": round(conf * 100, 2)})

    detected_count = len(filtered)

    # Update hasil
    with ESP_LOCK:
        ESP_RESULTS[esp_id] = {
            "kolam": kolam,
            "count": detected_count,
            "last_update": int(time.time() * 1000)
        }
        save_esp_results()

        # Backup ke GitHub
        try:
            content_b64 = base64.b64encode(json.dumps(ESP_RESULTS, indent=2).encode()).decode()
            get_res = requests.get(f"{GITHUB_API_ROOT}/esp_results.json", headers=GITHUB_HEADERS, timeout=10)
            sha = get_res.json().get("sha") if get_res.status_code == 200 else None
            put_data = {
                "message": "Update esp_results.json",
                "content": content_b64,
                "branch": "master"
            }
            if sha:
                put_data["sha"] = sha
            requests.put(f"{GITHUB_API_ROOT}/esp_results.json", headers=GITHUB_HEADERS, json=put_data, timeout=15)
        except Exception as e:
            print(f"[WARN] Backup JSON gagal: {e}")

        total_indukan = sum(ESP_RESULTS.get(esp, {}).get("count", 0) for esp in KOLAM_CONFIG["indukan"]["esp_ids"])
        total_anakan = sum(ESP_RESULTS.get(esp, {}).get("count", 0) for esp in KOLAM_CONFIG["anakan"]["esp_ids"])

    return jsonify({
        "status": "ok",
        "esp_id": esp_id,
        "kolam": kolam,
        "detected_this_upload": detected_count,
        "total_indukan": total_indukan,
        "total_anakan": total_anakan,
        "total_semua": total_indukan + total_anakan,
        "per_esp": ESP_RESULTS,
        "objects": filtered
    })

@app.route("/summary", methods=["GET"])
def summary():
    with ESP_LOCK:
        total_indukan = sum(ESP_RESULTS.get(esp, {}).get("count", 0) for esp in KOLAM_CONFIG["indukan"]["esp_ids"])
        total_anakan = sum(ESP_RESULTS.get(esp, {}).get("count", 0) for esp in KOLAM_CONFIG["anakan"]["esp_ids"])
        return jsonify({
            "total_indukan": total_indukan,
            "total_anakan": total_anakan,
            "total_semua": total_indukan + total_anakan,
            "per_esp": ESP_RESULTS,
            "last_updated": int(time.time() * 1000)
        })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)