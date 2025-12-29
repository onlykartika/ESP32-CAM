import os
import time
import base64
import requests
from flask import Flask, request, jsonify, send_from_directory
from inference_sdk import InferenceHTTPClient

# ================= FLASK APP =================
app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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

# ================= HEALTH =================
@app.route("/")
def health():
    return "Render AI server running"

# ================= STATIC IMAGE =================
@app.route("/uploads/<filename>")
def get_image(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# ================= IMAGE UPLOAD =================
@app.route("/upload", methods=["POST"])
def upload():
    if not request.data:
        return jsonify({"error": "no image received"}), 400

    timestamp = int(time.time())
    filename = f"{timestamp}.jpg"
    filepath = os.path.join(UPLOAD_FOLDER, filename)

    # ===== SIMPAN FILE =====
    with open(filepath, "wb") as f:
        f.write(request.data)

    # ===== ROBOFLOW =====
    try:
        result = client.run_workflow(
            workspace_name=WORKSPACE_NAME,
            workflow_id=WORKFLOW_ID,
            images={"image": filepath},
            use_cache=False
        )
    except Exception as e:
        return jsonify({"error": "roboflow failed", "detail": str(e)}), 500

    # ===== UPLOAD KE GITHUB =====
    try:
        with open(filepath, "rb") as f:
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

    # ===== PARSE HASIL =====
    predictions = []
    if isinstance(result, list):
        for item in result:
            if "predictions" in item:
                predictions.extend(item["predictions"])

    TARGET_LABEL = "panulirus ornatus - juvenile"
    filtered = [
        {
            "label": p.get("class"),
            "confidence": round(p.get("confidence", 0) * 100, 2)
        }
        for p in predictions
        if p.get("class", "").lower() == TARGET_LABEL.lower()
    ]

    # ===== RESPONSE =====
    image_url = request.host_url + "uploads/" + filename

    return jsonify({
        "status": "ok",
        "filename": filename,
        "image_url": image_url,
        "total_detected": len(filtered),
        "objects": filtered
    }), 200


# ================= RUN =================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
