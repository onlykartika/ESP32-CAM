import os
import time
import base64
import json
import requests
import tempfile
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
        print("[INFO] ESP_RESULTS saved")
    except Exception as e:
        print(f"[ERROR] Failed saving: {e}")

def load_esp_results():
    global ESP_RESULTS
    if os.path.exists(ESP_RESULTS_FILE):
        try:
            with open(ESP_RESULTS_FILE, "r") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    ESP_RESULTS = data
                    print("[INFO] ESP_RESULTS loaded from local")
                    return
        except Exception as e:
            print(f"[ERROR] Failed loading local: {e}")

    try:
        res = requests.get(
            f"{GITHUB_API_ROOT}/esp_results.json",
            headers=GITHUB_HEADERS, timeout=10
        )
        if res.status_code == 200:
            content = base64.b64decode(res.json()["content"]).decode("utf-8")
            data    = json.loads(content)
            if isinstance(data, dict):
                ESP_RESULTS = data
                save_esp_results()
                print("[INFO] ESP_RESULTS loaded from GitHub")
                return
    except Exception as e:
        print(f"[WARN] Failed loading from GitHub: {e}")

    ESP_RESULTS = {}
    print("[INFO] ESP_RESULTS initialized empty")

load_esp_results()

# ================= ROBOFLOW =================
WORKSPACE_NAME = "bocil-musik"
WORKFLOW_ID    = "detect-count-and-visualize"
TARGET_LABEL   = "juvenile"
CONF_THRESHOLD = 0.4

rf_client = None

def get_rf_client():
    global rf_client
    if rf_client is None:
        rf_client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key=ROBOFLOW_API_KEY
        )
    return rf_client

# ================= ROBOFLOW VIA REST API (fallback jika SDK gagal) =================
# Sama persis seperti Colab — kirim langsung via HTTP tanpa SDK
def run_roboflow_rest(image_bytes):
    """
    Kirim gambar ke Roboflow Workflow via REST API langsung.
    Ini identik dengan yang dipakai Colab di balik layar.
    """
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    url = f"https://serverless.roboflow.com/{WORKSPACE_NAME}/{WORKFLOW_ID}"
    headers = {
        "Content-Type": "application/json"
    }
    payload = {
        "api_key": ROBOFLOW_API_KEY,
        "inputs": {
            "image": {
                "type": "base64",
                "value": image_b64
            }
        }
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    print(f"[DEBUG] REST API status: {resp.status_code}")
    print(f"[DEBUG] REST API response: {resp.text[:500]}")

    if resp.status_code != 200:
        raise Exception(f"Roboflow REST error {resp.status_code}: {resp.text}")

    return resp.json()

# ================= PARSE PREDICTIONS =================
def parse_predictions(result):
    """Sama persis dengan logika Colab yang berhasil"""
    raw = None
    if isinstance(result, dict) and "predictions" in result:
        raw = result["predictions"]
    elif isinstance(result, list) and result and "predictions" in result[0]:
        raw = result[0]["predictions"]

    if isinstance(raw, list):
        return raw
    elif isinstance(raw, dict) and "predictions" in raw:
        return raw["predictions"]
    return []

# ================= HEALTH CHECK =================
@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status":          "ok",
        "target_label":    TARGET_LABEL,
        "conf_threshold":  CONF_THRESHOLD
    })

# ================= IMAGE UPLOAD =================
@app.route("/upload", methods=["POST"])
def upload():
    image_data = request.data
    if not image_data:
        return jsonify({"error": "no image received"}), 400

    image_size = len(image_data)
    print(f"[INFO] Received {image_size} bytes")

    if image_size < 1000:
        return jsonify({
            "error": f"gambar terlalu kecil ({image_size} bytes)",
            "hint":  "Thunder Client: tab Body > Binary > pilih file JPG"
        }), 400

    esp_id    = request.headers.get("X-ESP-ID", "unknown")
    timestamp = int(time.time())
    filename  = f"{esp_id}_{timestamp}.jpg"

    # Simpan ke file sementara
    try:
        with open(filename, "wb") as f:
            f.write(image_data)
        print(f"[INFO] Saved: {filename} ({image_size} bytes)")
    except Exception as e:
        return jsonify({"error": "failed to save", "detail": str(e)}), 500

    # ===== COBA SDK DULU, FALLBACK KE REST API =====
    result = None
    method_used = ""

    # Metode 1: SDK (sama seperti Colab)
    try:
        rf     = get_rf_client()
        result = rf.run_workflow(
            workspace_name=WORKSPACE_NAME,
            workflow_id=WORKFLOW_ID,
            images={"image": filename},
            use_cache=True
        )
        method_used = "SDK"
        print(f"[INFO] Roboflow SDK berhasil")
        try:
            print("[DEBUG] SDK result:\n" + json.dumps(result, indent=2, default=str))
        except Exception:
            print("[DEBUG] SDK result:", str(result)[:500])

    except Exception as sdk_err:
        print(f"[WARN] SDK gagal: {sdk_err} — mencoba REST API...")

        # Metode 2: REST API langsung (fallback)
        try:
            result      = run_roboflow_rest(image_data)
            method_used = "REST"
            print(f"[INFO] Roboflow REST berhasil")
        except Exception as rest_err:
            print(f"[ERROR] REST juga gagal: {rest_err}")
            if os.path.exists(filename):
                os.remove(filename)
            return jsonify({
                "error":      "roboflow failed",
                "sdk_error":  str(sdk_err),
                "rest_error": str(rest_err)
            }), 500

    # ===== UPLOAD GAMBAR KE GITHUB =====
    try:
        with open(filename, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()
        res = requests.put(
            f"{GITHUB_API_IMAGES}/{esp_id}/{filename}",
            headers=GITHUB_HEADERS,
            json={"message": f"upload from {esp_id}", "content": img_b64},
            timeout=15
        )
        print(f"[INFO] GitHub image: {res.status_code}")
    except Exception as e:
        print(f"[WARN] GitHub image error: {e}")

    if os.path.exists(filename):
        os.remove(filename)

    # ===== PARSE PREDICTIONS =====
    predictions_list = parse_predictions(result)

    print(f"[DEBUG] method={method_used}, total predictions={len(predictions_list)}")
    for p in predictions_list:
        lbl  = p.get("class") or p.get("label") or "unknown"
        conf = float(p.get("confidence") or p.get("score") or 0.0)
        print(f"   → '{lbl}' ({conf*100:.1f}%)")

    # Filter target
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

    # ===== SIMPAN HASIL =====
    with ESP_LOCK:
        ESP_RESULTS[esp_id] = {
            "count":       detected_count,
            "last_update": int(time.time() * 1000)
        }
        save_esp_results()

        try:
            json_content = json.dumps(ESP_RESULTS).encode("utf-8")
            content_b64  = base64.b64encode(json_content).decode("utf-8")

            get_res = requests.get(
                f"{GITHUB_API_ROOT}/esp_results.json",
                headers=GITHUB_HEADERS, timeout=10
            )
            sha = get_res.json().get("sha") if get_res.status_code == 200 else None

            put_data = {
                "message": f"Update from {esp_id} at {time.strftime('%Y-%m-%d %H:%M:%S')}",
                "content": content_b64
            }
            if sha:
                put_data["sha"] = sha

            put_res = requests.put(
                f"{GITHUB_API_ROOT}/esp_results.json",
                headers=GITHUB_HEADERS, json=put_data, timeout=15
            )
            print(f"[INFO] GitHub JSON: {put_res.status_code}")
        except Exception as e:
            print(f"[WARN] GitHub JSON error: {e}")

    total_all = sum(v["count"] for v in ESP_RESULTS.values())

    return jsonify({
        "status":                 "ok",
        "esp_id":                 esp_id,
        "method_used":            method_used,
        "image_size_bytes":       image_size,
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
