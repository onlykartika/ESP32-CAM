import os
import time
import base64
import requests
from flask import Flask, request, jsonify
from inference_sdk import InferenceHTTPClient

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

# ================= HEALTH CHECK =================
@app.route("/", methods=["GET"])
def health():
    return "Render AI server running"

# ================= IMAGE UPLOAD =================
@app.route("/upload", methods=["POST"])
def upload():
    if not request.data:
        return jsonify({"error": "no image received"}), 400

    # ===== SIMPAN IMAGE SEMENTARA =====
    timestamp = int(time.time())
    filename = f"{timestamp}.jpg"

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

    # ===== UPLOAD IMAGE KE GITHUB =====
    try:
        with open(filename, "rb") as f:
            content = base64.b64encode(f.read()).decode()

        requests.put(
            f"{GITHUB_API}/{filename}",
            headers=GITHUB_HEADERS,
            json={
                "message": f"upload from esp32 ({filename})",
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
            filtered.append({
                "label": label,
                "confidence": round(conf * 100, 2)
            })

    # ===== RESPONSE JSON =====
    response = {
        "status": "ok",
        "filename": filename,
        "total_detected": len(filtered),
        "objects": filtered
    }

    return jsonify(response), 200


# ================= RUN LOCAL (Render pakai gunicorn) =================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)