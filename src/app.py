from fastapi import FastAPI
from train.train_util import load_model
import uvicorn
import io
import numpy as np
from fastapi import FastAPI, UploadFile, File
from PIL import Image
from evaluate.evaluate import get_prediction

# Load model
loaded_cnn=load_model("cnn_model.pkl")

# -------------------------------------------------
# App initialization
# -------------------------------------------------
app = FastAPI(
    title="Cats and Dogs Image Classification API",
    description="Predict cat ot dog image using ML models",
    version="1.0.0"
)

# -------------------------------------------------
# Health check endpoint
# -------------------------------------------------
@app.get("/")
def health():
    return {"status": "ok"}


def preprocess_image(image_bytes: bytes):
    # Convert bytes to PIL Image
    img = Image.open(io.BytesIO(image_bytes))
    
    # Ensure RGB (converts RGBA/Grayscale to 3-channel)
    img = img.convert("RGB")
    
    # Resize to match training input
    img = img.resize((224, 224))
    
    # Convert to array and add batch dimension
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)

    
    return img_array

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read the uploaded file
    contents = await file.read()
    
    # Preprocess
    processed_img = preprocess_image(contents)
    
    # Execute model inference
    # prediction = loaded_cnn.predict(processed_img)[0][0]


    pred_label, conf, prob = get_prediction(loaded_cnn,"",processed_img)

    return {
        "prediction": pred_label,
        "confidence": round(conf * 100, 2),
        "raw_probability": float(prob)
    }


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000
    )
