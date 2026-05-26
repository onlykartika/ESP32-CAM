import os
import time
import base64
import json
import requests
from flask import Flask, request, jsonify
from inference_sdk import InferenceHTTPClient
from threading import Lock

app = Flask(__name__)

# ================= ENV =================
ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY")
GITHUB_TOKEN     = os.environ.get("GITHUB_TOKEN")
if not ROBOFLOW_API_KEY:
    raise ValueError("ROBOFLOW_API_KEY environment variable is required")
if not GITHUB_TOKEN:
    raise ValueError("GITHUB_TOKEN environment variable is required")

# ================= PERSISTENT STORAGE =================
ESP_RESULTS_FILE = "esp_results.json"
ESP_RESULTS      = {}
ESP_LOCK         = Lock()

# ================= GITHUB CONFIG =================
GITHUB_REPO       = "onlykartika/ESP32-CAM"
GITHUB_FOLDER     = "images"
GITHUB_API_IMAGES = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{GITHUB_FOLDER}"
GITHUB_API_ROOT   = f"https://api.github.com/repos/{GITHUB_REPO}/contents"
GITHUB_HEADERS    = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "User-Agent":    "Render-AI-Server",
    "Accept":        "application/vnd.github.v3+json"
}

# ================= SAVE / LOAD =================
def save_esp_results():
    try:
        with open(ESP_RESULTS_FILE, "w") as f:
            json.dump(ESP_RESULTS, f, indent=2)
        print("[INFO] ESP_RESULTS saved to local file")
    except Exception as e:
        print(f"[ERROR] Failed saving local esp_results.json: {e}")

def load_esp_results():
    global ESP_RESULTS
    if os.path.exists(ESP_RESULTS_FILE):
        try:
            with open(ESP_RESULTS_FILE, "r") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    ESP_RESULTS = data
                    print("[INFO] ESP_RESULTS loaded from local file")
                    return
        except Exception as e:
            print(f"[ERROR] Failed loading local esp_results.json: {e}")

    try:
        get_url = f"{GITHUB_API_ROOT}/esp_results.json"
        res = requests.get(get_url, headers=GITHUB_HEADERS, timeout=10)
        if res.status_code == 200:
            content_b64 = res.json()["content"]
            content     = base64.b64decode(content_b64).decode("utf-8")
            data        = json.loads(content)
            if isinstance(data, dict):
                ESP_RESULTS = data
                save_esp_results()
                print("[INFO] ESP_RESULTS loaded from GitHub fallback")
                return
        else:
            print(f"[WARN] esp_results.json not found on GitHub (status {res.status_code})")
    except Exception as e:
        print(f"[WARN] Failed loading from GitHub: {e}")

    ESP_RESULTS = {}
    print("[INFO] ESP_RESULTS initialized as empty")

load_esp_results()

# ================= ROBOFLOW =================
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
WORKFLOW_ID    = "detect-count-and-visualize"
TARGET_LABEL   = "panulirus ornatus - juvenile"  # sesuaikan dengan label di model kamu
CONF_THRESHOLD = 0.4

# ================= HELPER: PARSE PREDICTIONS =================
# Logika SAMA PERSIS dengan Colab yang berhasil
def parse_predictions(result):
    raw = None
    if isinstance(result, dict) and "predictions" in result:
        raw = result["predictions"]
    elif isinstance(result, list) and result and "predictions" in result[0]:
        raw = result[0]["predictions"]

    predictions_list = []
    if isinstance(raw, list):
        predictions_list = raw
    elif isinstance(raw, dict) and "predictions" in raw:
        predictions_list = raw["predictions"]

    return predictions_list

# ================= HEALTH CHECK =================
@app.route("/", methods=["GET"])
def health():
    return "Render AI server running"

# ================= IMAGE UPLOAD =================
@app.route("/upload", methods=["POST"])
def upload():
    if not request.data:
        return jsonify({"error": "no image received"}), 400

    esp_id    = request.headers.get("X-ESP-ID", "unknown")
    timestamp = int(time.time())
    filename  = f"{esp_id}_{timestamp}.jpg"

    print(f"[INFO] Upload dari ESP: {esp_id}, ukuran: {len(request.data)} bytes")

    # ===== SIMPAN GAMBAR KE DISK =====
    # KUNCI: simpan dulu ke file, lalu kirim file path ke Roboflow
    # persis seperti di Colab (bukan kirim bytes langsung)
    try:
        with open(filename, "wb") as f:
            f.write(request.data)
        print(f"[INFO] Gambar disimpan: {filename}")
    except Exception as e:
        return jsonify({"error": "failed to save image", "detail": str(e)}), 500

    # ===== ROBOFLOW WORKFLOW =====
    result = None
    try:
        rf     = get_rf_client()

        # Kirim FILE PATH — sama seperti Colab yang berhasil
        result = rf.run_workflow(
            workspace_name=WORKSPACE_NAME,
            workflow_id=WORKFLOW_ID,
            images={"image": filename},  # file path, bukan bytes
            use_cache=True
        )

        print("[INFO] Roboflow selesai")
        try:
            print("[DEBUG] Raw result:\n" + json.dumps(result, indent=2, default=str))
        except Exception:
            print("[DEBUG] Raw result:", str(result))

    except Exception as e:
        print(f"[ERROR] Roboflow gagal: {e}")
        if os.path.exists(filename):
            os.remove(filename)
        return jsonify({"error": "roboflow failed", "detail": str(e)}), 500

    # ===== UPLOAD GAMBAR KE GITHUB =====
    try:
        with open(filename, "rb") as f:
            img_content = base64.b64encode(f.read()).decode()
        put_url = f"{GITHUB_API_IMAGES}/{esp_id}/{filename}"
        res = requests.put(
            put_url,
            headers=GITHUB_HEADERS,
            json={
                "message": f"upload from {esp_id} ({filename})",
                "content": img_content
            },
            timeout=15
        )
        if res.status_code in (200, 201):
            print(f"[INFO] Gambar {filename} berhasil ke GitHub")
        else:
            print(f"[WARN] GitHub upload gagal: {res.status_code} - {res.text}")
    except Exception as e:
        print(f"[WARN] GitHub upload error: {e}")

    # Hapus file sementara
    if os.path.exists(filename):
        os.remove(filename)

    # ===== PARSE PREDICTIONS — logika sama dengan Colab =====
    predictions_list = parse_predictions(result)

    # Debug: print semua label yang terdeteksi
    print(f"[DEBUG] Total predictions: {len(predictions_list)}")
    for p in predictions_list:
        lbl  = p.get("class") or p.get("label") or "unknown"
        conf = p.get("confidence") or p.get("score") or 0.0
        print(f"   → {lbl} ({float(conf)*100:.1f}%)")

    # Filter sesuai target
    filtered = []
    for p in predictions_list:
        label = p.get("class") or p.get("label") or "unknown"
        conf  = float(p.get("confidence") or p.get("score") or 0.0)
        if label.lower() == TARGET_LABEL.lower() and conf >= CONF_THRESHOLD:
            filtered.append({
                "label":      label,
                "confidence": round(conf * 100, 2)
            })

    detected_count = len(filtered)
    print(f"[INFO] '{TARGET_LABEL}' terdeteksi: {detected_count}")

    # ===== UPDATE & SAVE RESULT =====
    with ESP_LOCK:
        ESP_RESULTS[esp_id] = {
            "count":       detected_count,
            "last_update": int(time.time() * 1000)
        }
        save_esp_results()

        # Push ke GitHub
        try:
            json_content = json.dumps(ESP_RESULTS).encode("utf-8")
            content_b64  = base64.b64encode(json_content).decode("utf-8")

            get_url = f"{GITHUB_API_ROOT}/esp_results.json"
            get_res = requests.get(get_url, headers=GITHUB_HEADERS, timeout=10)
            sha     = get_res.json().get("sha") if get_res.status_code == 200 else None

            put_data = {
                "message": f"Update results from {esp_id} at {time.strftime('%Y-%m-%d %H:%M:%S')}",
                "content": content_b64
            }
            if sha:
                put_data["sha"] = sha

            put_res = requests.put(
                f"{GITHUB_API_ROOT}/esp_results.json",
                headers=GITHUB_HEADERS,
                json=put_data,
                timeout=15
            )
            if put_res.status_code in (200, 201):
                print("[INFO] esp_results.json berhasil ke GitHub")
            else:
                print(f"[WARN] GitHub JSON upload gagal: {put_res.status_code} {put_res.text}")
        except Exception as e:
            print(f"[WARN] GitHub JSON upload error: {e}")

    total_all = sum(v["count"] for v in ESP_RESULTS.values())

    return jsonify({
        "status":                 "ok",
        "esp_id":                 esp_id,
        "detected_this_esp":      detected_count,
        "total_detected_all_esp": total_all,
        "per_esp":                ESP_RESULTS,
        "objects":                filtered
    }), 200

# ================= SUMMARY =================
@app.route("/summary", methods=["GET"])
def summary():
    with ESP_LOCK:
        return jsonify({
            "total_all_esp": sum(v["count"] for v in ESP_RESULTS.values()),
            "per_esp":       ESP_RESULTS
        })

# ================= RUN =================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
