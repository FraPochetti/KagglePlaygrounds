import json
import logging
import sys
import os
import io

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def model_fn(model_dir): 
    model = torch.jit.load(os.path.join(model_dir, "model.pth")).eval().to(device)
    return model   


def input_fn(request_body, request_content_type):

    f = io.BytesIO(request_body)
    input_image = Image.open(f).convert("RGB")
    preprocess = transforms.Compose(
        [
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)
    return input_batch


def predict_fn(input_object, model):
    with torch.no_grad():
        prediction = model(input_object.to(device))
    return prediction


def output_fn(predictions, content_type):
    assert content_type == "application/json"
    res = predictions.cpu().numpy().tolist()
    return json.dumps(res)
