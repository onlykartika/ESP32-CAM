from inference_sdk import InferenceHTTPClient

# ===== ROBOFLOW CONFIG =====
API_KEY = "1l1sKsEAOcWMo4yBVwZQ"
WORKSPACE_NAME = "my-workspace-grzes"
WORKFLOW_ID = "detect-count-and-visualize"
TARGET = "panulirus ornatus - juvenile"

# ===== IMAGE URL DARI GITHUB =====
IMAGES = {
    "esp1": "https://raw.githubusercontent.com/kartika/esp32-images/main/esp1.jpg",
    "esp2": "https://raw.githubusercontent.com/kartika/esp32-images/main/esp2.jpg",
    "esp3": "https://raw.githubusercontent.com/kartika/esp32-images/main/esp3.jpg",
    "esp4": "https://raw.githubusercontent.com/kartika/esp32-images/main/esp3.jpg",
}

# ===== CLIENT =====
client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=API_KEY
)

# ===== LOOP =====
for name, url in IMAGES.items():
    print(f"Processing {name} ...")

    result = client.run_workflow(
        workspace_name=WORKSPACE_NAME,
        workflow_id=WORKFLOW_ID,
        images={"image": url},
        use_cache=False
    )

    # ===== PARSE =====
    predictions = []
    if isinstance(result, dict) and "predictions" in result:
        predictions = result["predictions"]

    filtered = []
    for p in predictions:
        label = p.get("class") or p.get("label")
        conf = p.get("confidence") or p.get("score") or 0
        if label and label.lower() == TARGET.lower():
            filtered.append((label, conf))

    # ===== OUTPUT =====
    print(f"HASIL {name}")
    print("Total:", len(filtered))
    for i, (label, conf) in enumerate(filtered, 1):
        print(f"{i}. {label} ({conf*100:.1f}%)")
    print("-"*40)
