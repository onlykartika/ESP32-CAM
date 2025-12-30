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

# ================= GITHUB =================
GITHUB_REPO = "onlykartika/ESP32-CAM"
GITHUB_FOLDER = "images"
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
                print("[INFO] Loaded local esp_results.json")
        except:
            ESP_RESULTS = {}

    try:
        res = requests.get(
            f"{GITHUB_API_ROOT}/esp_results.json",
            headers=GITHUB_HEADERS,
            timeout=10
        )
        if res.status_code == 200:
            content = base64.b64decode(res.json()["content"]).decode()
            ESP_RESULTS = json.loads(content)
            save_esp_results()
            print("[INFO] Loaded esp_results.json from GitHub")
    except Exception as e:
        print("[WARN] GitHub load failed:", e)

    if not isinstance(ESP_RESULTS, dict):
        ESP_RESULTS = {}

def save_esp_results():
    with open(ESP_RESULTS_FILE, "w") as f:
        json.dump(ESP_RESULTS, f, indent=2)

load_esp_results()

# ================= ROBOFLOW CLIENT (DUA CLIENT) =================
rf_client_indukan = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=ROBOFLOW_API_KEY_INDUKAN
)

rf_client_anakan = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=ROBOFLOW_API_KEY_ANAKAN
)

# ================= KOLAM CONFIG =================
KOLAM_CONFIG = {
    "indukan": {
        "esp_ids": ["esp_5", "esp_6", "esp_7", "esp_8"],
        "client": rf_client_indukan,
        "workspace": "my-workspace-rrwxa",
        "workflow": "detect-count-and-visualize",
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
    return "AI Server – Dual Roboflow API – Running"

@app.route("/upload", methods=["POST"])
def upload():
    if not request.data:
        return jsonify({"error": "no image"}), 400

    esp_id = request.headers.get("X-ESP-ID", "unknown").lower()
    print(f"[INFO] Upload dari {esp_id}")

    # ================= PILIH KOLAM =================
    kolam = None
    for k, cfg in KOLAM_CONFIG.items():
        if esp_id in cfg["esp_ids"]:
            kolam = k
            break

    if not kolam:
        return jsonify({"error": "esp_id not registered"}), 400

    cfg = KOLAM_CONFIG[kolam]

    # ================= SIMPAN IMAGE =================
    ts = int(time.time())
    filename = f"{esp_id}_{ts}.jpg"
    with open(filename, "wb") as f:
        f.write(request.data)

    # ================= ROBOFLOW (FIXED) =================
    try:
        with open(filename, "rb") as img:
            image_bytes = img.read()

        result = cfg["client"].run_workflow(
            workspace_name=cfg["workspace"],
            workflow_id=cfg["workflow"],
            images={"image": image_bytes},  # ✅ WAJIB BYTES
            use_cache=False
        )
    except Exception as e:
        os.remove(filename)
        return jsonify({
            "error": "roboflow failed",
            "detail": str(e)
        }), 500

    # ================= PARSE RESULT =================
    predictions = []

    if isinstance(result, list):
        for r in result:
            predictions.extend(r.get("predictions", []))
    elif isinstance(result, dict):
        predictions = result.get("predictions", [])

    detected = []
    for p in predictions:
        label = p.get("class") or p.get("label")
        conf = p.get("confidence") or p.get("score") or 0
        if (
            label
            and label.lower() == cfg["target_label"].lower()
            and conf >= cfg["conf_threshold"]
        ):
            detected.append({
                "label": label,
                "confidence": round(conf * 100, 2)
            })

    count = len(detected)

    # ================= UPLOAD IMAGE TO GITHUB =================
    try:
        with open(filename, "rb") as f:
            content_b64 = base64.b64encode(f.read()).decode()

        put_url = f"{GITHUB_API_ROOT}/{GITHUB_FOLDER}/{cfg['folder']}/{filename}"

        requests.put(
            put_url,
            headers=GITHUB_HEADERS,
            json={
                "message": f"upload {filename}",
                "content": content_b64
            },
            timeout=15
        )
    except Exception as e:
        print("[WARN] GitHub upload failed:", e)

    os.remove(filename)

    # ================= UPDATE JSON =================
    with ESP_LOCK:
        ESP_RESULTS[esp_id] = {
            "count": count,
            "kolam": kolam,
            "last_update": int(time.time() * 1000)
        }
        save_esp_results()

        try:
            json_b64 = base64.b64encode(
                json.dumps(ESP_RESULTS, indent=2).encode()
            ).decode()

            get_res = requests.get(
                f"{GITHUB_API_ROOT}/esp_results.json",
                headers=GITHUB_HEADERS
            )
            sha = get_res.json().get("sha") if get_res.status_code == 200 else None

            payload = {
                "message": "update esp_results.json",
                "content": json_b64
            }
            if sha:
                payload["sha"] = sha

            requests.put(
                f"{GITHUB_API_ROOT}/esp_results.json",
                headers=GITHUB_HEADERS,
                json=payload
            )
        except Exception as e:
            print("[WARN] GitHub JSON update failed:", e)

    return jsonify({
        "status": "ok",
        "esp_id": esp_id,
        "kolam": kolam,
        "detected": count,
        "objects": detected
    }), 200

@app.route("/summary", methods=["GET"])
def summary():
    with ESP_LOCK:
        return jsonify(ESP_RESULTS)
