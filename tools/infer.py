from fastapi import FastAPI, File, UploadFile
import numpy as np

app = FastAPI()

# Dummy model function (Replace with actual model inference)
def process_stereo_images(left_image, right_image):
    """
    Mock function to simulate stereo image processing.
    Replace this with actual model inference.
    """
    return [
        {"x_min": 50, "y_min": 60, "x_max": 200, "y_max": 250, "confidence": 0.95, "class": "car"},
        {"x_min": 300, "y_min": 100, "x_max": 400, "y_max": 200, "confidence": 0.88, "class": "pedestrian"}
    ]

@app.get('/')
def health_check():
    return {"status": "ok"}

@app.post("/predict/")
async def predict(left_image: UploadFile = File(...), right_image: UploadFile = File(...)):
    """
    API endpoint to receive stereo image pairs and return bounding box predictions.
    """

    # Process with model
    predictions = process_stereo_images(left_image, right_image)

    return {"bounding_boxes": predictions}