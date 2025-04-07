from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException
from pathlib import Path

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils, box_utils
from pcdet.datasets import build_dataloader

import torch
import time

# Upload folder
UPLOAD_FOLDER = Path("data/kitti_tracking/training")
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

# Global model and dataset variables
model, test_set, test_loader = None, None, None

def load_model():
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


def infer(model, batch_dict):
    load_data_to_gpu(batch_dict)

    with torch.no_grad():
        pred_dicts, _ = model(batch_dict)

    calib = batch_dict['calib'][0]
    pred_boxes_cam = box_utils.boxes3d_lidar_to_kitti_camera(pred_dicts[0]['pred_boxes'].cpu().numpy(), calib)

    return pred_boxes_cam.tolist()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, test_loader, test_set
    print("Loading model...")
    load_model()
    print("Model loaded successfully!")
    yield  # Run the app
    print("Shutting down...")

app = FastAPI(lifespan=lifespan)

@app.get("/")
def root():
    return {"message": "Stereo Object Detection API is running!"}

@app.post("/predict")
async def predict(left_image: UploadFile = File(...), right_image: UploadFile = File(...), calib: UploadFile = File(...)):
    """
    API endpoint to receive stereo image pairs and return bounding box predictions.
    """

    try:
        global model, test_loader

        img_l_path = UPLOAD_FOLDER / "image_02/0/0.png"
        img_r_path = UPLOAD_FOLDER / "image_03/0/0.png"
        calib_path = UPLOAD_FOLDER / "calib/0.txt"

        with img_l_path.open("wb") as f:
            f.write(await left_image.read())
        with img_r_path.open("wb") as f:
            f.write(await right_image.read())
        with calib_path.open("wb") as f:
            f.write(await calib.read())

        batch_dict = test_loader[0]

        pred_boxes = infer(model, batch_dict)
        return {"bounding_boxes": pred_boxes}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))