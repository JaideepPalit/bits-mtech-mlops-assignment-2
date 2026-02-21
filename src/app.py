import time
import io
import numpy as np
import uvicorn
from fastapi import FastAPI, UploadFile, File
from PIL import Image

# Mocking internal imports for the example
from train.train_util import load_model
from evaluate.evaluate import get_prediction

# Load model
loaded_cnn = load_model("cnn_model.pkl")

app = FastAPI(
    title="Cats and Dogs Image Classification API",
    version="1.0.0"
)

# -------------------------------------------------
# Metric Storage (Specific to Predict)
# -------------------------------------------------
app.state.predict_count = 0
app.state.predict_total_latency = 0.0

# -------------------------------------
# Endpoints
# -------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/metrics")
def get_metrics():
    # Calculate average only for the predict endpoint
    avg_latency = (
        app.state.predict_total_latency / app.state.predict_count 
        if app.state.predict_count > 0 else 0
    )
    return {
        "endpoint": "/predict",
        "request_count": app.state.predict_count,
        "total_latency_seconds": round(app.state.predict_total_latency, 4),
        "average_latency_seconds": round(avg_latency, 4)
    }

def preprocess_image(image_bytes: bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img)
    return np.expand_dims(img_array, axis=0)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # --- Start Timer ---
    start_time = time.perf_counter()
    
    # Read and process
    contents = await file.read()
    processed_img = preprocess_image(contents)
    
    # Run Inference
    pred_label, conf, prob = get_prediction(loaded_cnn, "", processed_img)

    # --- End Timer and Update Stats ---
    latency = time.perf_counter() - start_time
    app.state.predict_count += 1
    app.state.predict_total_latency += latency

    return {
        "prediction": pred_label,
        "confidence": round(conf * 100, 2),
        "raw_probability": float(prob)
    }

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)