from loguru import logger
import uvicorn
from fastapi import (
    FastAPI,
    File,
    UploadFile,
)
from pydantic import (
    BaseModel,
    Field,
)
from icevision import models
import torch
import time
from inference_utils import *

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
logger.info(f"Running inference on: {device}")

model_type = models.ultralytics.yolov5
backbone = model_type.backbones.medium
model = model_type.model(backbone=backbone(pretrained=True), num_classes=2, img_size=IMAGE_SIZE)
model.load_state_dict(torch.load("model.pth"))
model = model.to(device)
logger.info(f"Model loaded on: {device}")

app = FastAPI()

class OcrResponse(BaseModel):
    text: str = Field(..., example="ABC 123")
    confidence: float = Field(..., example=0.99)
    bbox_width: float = Field(..., example=0.5)
    bbox_height: float = Field(..., example=0.5)
    bbox_area: float = Field(..., example=0.8)

@app.get("/", summary="Application health check")
def health_check() -> Dict[str, str]:
    return {"Status": "Application up and running."}

@app.post(
    "/predict",
    summary="Reads the car license plate from photo.",
    response_model=OcrResponse,
)
def upload_image(
    image: UploadFile = File(...),
):
    s = time.time()
    image_file = image.file.read()
    pilimage = read_imagefile(image_file)

    logger.info(f"Read image in {time.time() - s:.2f} seconds")
    s = time.time()
    pred_dict  = model_type.end2end_detect(pilimage, INFER_TFMS, model, class_map=CLASS_MAP, detection_threshold=0.3)

    logger.info(f"Model inference done in {time.time() - s:.2f} seconds")
    s = time.time()

    bbox = extract_biggest_bbox(pred_dict["detection"]["bboxes"])
    coords = extract_coords_from_bbox(bbox)
    roi = crop_and_enhance(pilimage, coords)
    result = read_text_from_roi(roi, TEXTRACT)

    logger.info(f"Results post-processing in {time.time() - s:.2f} seconds")

    return result[0]

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0")
