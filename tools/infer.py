import os
import time
from pathlib import Path
from contextlib import asynccontextmanager

import cv2
import torch
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils, box_utils
from pcdet.datasets import build_dataloader

# Constants
UPLOAD_FOLDER = Path("data/kitti_tracking/training")
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

# Global model and dataset variables
model, test_set, test_loader = None, None, None


def load_model():
    """Load the detection model and initialize the test dataset."""
    global model, test_set, test_loader

    cfg_file = "configs/stream/kitti_models/stream_dsgn_r18-token_prev_next-feature_align_avg_fusion-lka_7-mcl_infer.yaml"
    ckpt_file = "extra_data/checkpoint_epoch_20.pth"

    cfg_from_yaml_file(cfg_file, cfg)
    cfg.TAG = Path(cfg_file).stem

    logger = common_utils.create_logger()

    test_set, test_loader, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        training=False,
        dist=False,
        workers=2,
        logger=logger
    )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    model.load_params_from_file(filename=ckpt_file, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()


def infer(model, data_dict):
    """Run inference on the given data."""
    load_data_to_gpu(data_dict)

    with torch.no_grad():
        pred_dicts, _ = model(data_dict)

    calib = data_dict['token']['calib'][0]
    pred_boxes_cam = box_utils.boxes3d_lidar_to_kitti_camera(
        pred_dicts[0]['pred_boxes'].cpu().numpy(), calib
    )

    class_names = cfg.CLASS_NAMES
    pred_scores = pred_dicts[0]['pred_scores'].cpu().numpy()
    pred_labels = pred_dicts[0]['pred_labels'].cpu().numpy()

    result = []
    for box, score, label in zip(pred_boxes_cam, pred_scores, pred_labels):
        result.append({
            "object_type": class_names[label - 1],
            "location": box[0:3].tolist(),  # x, y, z
            "dimensions": box[3:6].tolist(),  # h, w, l
            "rotation_y": float(box[6]),
            "score": float(score)
        })

    return {
        "objects": result,
        "boxes": pred_boxes_cam.tolist()
        }

def visualize_boxes(image_path, pred_boxes, calib, save_path='output/infer/result.png'):
    """
    Visualize predicted 3D bounding boxes on the input image.

    Args:
        image_path (str): Path to the input image.
        pred_boxes (np.ndarray): Predicted 3D bounding boxes (N, 7).
        calib (object): Calibration object with `corners3d_to_img_boxes` method.
        save_path (str): Where to save the output image.

    Returns:
        np.ndarray: Image with predicted 3D bounding boxes drawn.
    """
    transform_start_time = time.time_ns()

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Normalize and prepare image tensor
    img_tensor = torch.tensor(img / 255., dtype=torch.float32).permute(2, 0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_tensor = (img_tensor - mean) / std

    # De-normalize for visualization
    img_vis = (img_tensor * std + mean).clamp(0, 1)
    img_vis = (img_vis.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    # Project 3D boxes to 2D
    pred_box_corners = box_utils.boxes3d_to_corners3d_kitti_camera(pred_boxes)
    _, pred_box_corners_img = calib.corners3d_to_img_boxes(pred_box_corners)

    def draw_3d_box(img, corners_img, color=(255, 0, 0)):
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7)
        ]
        for edge in edges:
            pt1 = tuple(map(int, corners_img[edge[0]]))
            pt2 = tuple(map(int, corners_img[edge[1]]))
            cv2.line(img, pt1, pt2, color=color, thickness=2)

    for corners in pred_box_corners_img:
        draw_3d_box(img_vis, corners)

    # Save the visualized result
    img_bgr = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, img_bgr)

    print(f'Transform time: {(time.time_ns() - transform_start_time) / 1e9:.3f}s')
    print(f'Saved to {save_path}')

    return img_bgr


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context: load model at startup, release resources at shutdown."""
    global model, test_loader, test_set
    print("Loading model...")
    load_model()
    print("Model loaded successfully!")
    yield
    print("Shutting down...")


app = FastAPI(lifespan=lifespan)


@app.get("/")
def root():
    """Root endpoint to verify API is running."""
    return {"message": "Stereo Object Detection API is running!"}


@app.post("/predict")
async def predict(
        left_image: UploadFile = File(...),
        right_image: UploadFile = File(...),
        calib: UploadFile = File(...)
):
    """
    Predict 3D bounding boxes from uploaded stereo images and calibration file.

    Args:
        left_image (UploadFile): Left stereo image.
        right_image (UploadFile): Right stereo image.
        calib (UploadFile): Calibration file.

    Returns:
        prediction: JSON object containing detected objects and their bounding boxes.
    """
    try:
        global model, test_loader

        # Prepare upload paths
        img_l_path = UPLOAD_FOLDER / "image_02/0/0.png"
        img_r_path = UPLOAD_FOLDER / "image_03/0/0.png"
        calib_path = UPLOAD_FOLDER / "calib/0.txt"

        img_l_path.parent.mkdir(parents=True, exist_ok=True)
        img_r_path.parent.mkdir(parents=True, exist_ok=True)
        calib_path.parent.mkdir(parents=True, exist_ok=True)

        # Upload timing
        upload_start_time = time.time()

        with img_l_path.open("wb") as f:
            f.write(await left_image.read())
        with img_r_path.open("wb") as f:
            f.write(await right_image.read())
        with calib_path.open("wb") as f:
            f.write(await calib.read())

        upload_duration = time.time() - upload_start_time
        print(f"Upload time: {upload_duration:.3f} seconds")

        # Prepare input data
        data_dict = test_set[0]
        data_dict = test_set.collate_batch([data_dict])

        # Inference timing
        inference_start_time = time.time()

        prediction = infer(model, data_dict)

        inference_duration = time.time() - inference_start_time
        print(f"Inference time: {inference_duration:.3f} seconds")

        return { "objects": prediction['objects'], "boxes": prediction['boxes'] }
    except Exception as e:
        print(f"Error during inference: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    load_model()

    data_dict = test_set[0]
    data_dict = test_set.collate_batch([data_dict])

    prediction = infer(model, data_dict)

    visualize_boxes(
        str(UPLOAD_FOLDER / "image_02/0/0.png"),
        prediction['boxes'],
        data_dict['token']['calib'][0],
        save_path='output/infer/result.png'
    )

    print(f"Objects detected: {len(prediction['objects'])}")