from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import os
import shutil
from PIL import Image
import io

# Define the path to your saved model
MODEL_PATH = "models\TumerDetection.h5"
MODEL_LOADED = False # Flag to check if model is successfully loaded
model = None # Initialize model as None

# --- Load the trained model globally when the app starts ---
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Model '{MODEL_PATH}' loaded successfully.")
    model.summary()
    MODEL_LOADED = True
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure 'TumerDetection.h5' is in the same directory as main.py")
    # In a real-world scenario, you might want to log this error and potentially
    # halt the application if the model is crucial. For now, we'll set a flag.

# Initialize FastAPI app
app = FastAPI(
    title="Brain Tumor Detection API",
    description="API for detecting brain tumors from MRI images.",
    version="1.0.0"
)

# Mount the static files directory
# This allows your browser to access index.html and other static assets
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- Function to preprocess an image for prediction ---
def preprocess_image(image_bytes: bytes):
    """
    Loads, resizes, and prepares an image for model prediction.
    Assumes input bytes are image data.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB") # Ensure 3 channels
        img = img.resize((128, 128))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        # The model's first layer is Rescaling(1./255), so no need to normalize here
        return img_array
    except Exception as e:
        raise ValueError(f"Error processing image: {e}")

# --- Function to make a prediction ---
def predict_tumor(image_bytes: bytes):
    """
    Takes image bytes, preprocesses it, and returns the prediction result.
    """
    if not MODEL_LOADED or model is None:
        raise RuntimeError("Model is not loaded. Cannot make predictions.")

    preprocessed_img = preprocess_image(image_bytes)
    prediction = model.predict(preprocessed_img)
    # The output is a sigmoid activation, so values are between 0 and 1
    # You might want to set a threshold, e.g., > 0.5 for a "yes"
    if prediction[0][0] > 0.5:
        return "Tumor detected"
    else:
        return "No tumor detected"

# --- API Endpoints ---

@app.get("/", response_class=HTMLResponse, summary="Home page for image upload")
async def read_root():
    """
    Serves the main HTML page for uploading images.
    """
    return HTMLResponse(content=open("static/index.html").read(), status_code=200)

@app.post("/predict/", summary="Predict tumor presence from an uploaded image")
async def predict_upload(file: UploadFile = File(...)):
    """
    Receives an uploaded image file, processes it, and returns a tumor prediction.
    """
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Model is not loaded. Please check server logs.")

    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Invalid file type. Only image files are allowed.")

    try:
        # Read image file bytes
        image_bytes = await file.read()
        prediction_result = predict_tumor(image_bytes)
        return JSONResponse(content={"prediction": prediction_result}, status_code=200)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except RuntimeError as re:
        raise HTTPException(status_code=500, detail=str(re))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

# If you want to add a simple health check endpoint
@app.get("/health", summary="Check API health and model status")
async def health_check():
    """
    Checks if the API is running and if the model was loaded successfully.
    """
    status = "healthy" if MODEL_LOADED else "model_load_failed"
    return {"status": status, "model_loaded": MODEL_LOADED}