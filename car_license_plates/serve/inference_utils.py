from typing import List, Union, Tuple, Dict
import numpy as np
import cv2
from PIL import ImageEnhance, Image
from icevision import ClassMap, tfms, BBox
from botocore.client import Config
import boto3
from io import BytesIO

TEXTRACT = boto3.client('textract', region_name="eu-west-1", config=Config(retries = {'max_attempts': 20, 'mode': 'adaptive'}))
IMAGE_SIZE = 384
CLASS_MAP = ClassMap(['licence'])
INFER_TFMS = tfms.A.Adapter([*tfms.A.resize_and_pad(IMAGE_SIZE), tfms.A.Normalize()])

def read_imagefile(file: bytes) -> Image.Image:
    image = Image.open(BytesIO(file)).convert("RGB")
    return image

def image_to_bts(frame: np.array) -> bytes:
    '''
    :param frame: WxHx3 ndarray
    '''
    _, bts = cv2.imencode('.jpg', frame)
    bts = bts.tobytes()
    return bts

def crop_and_enhance(img: Image.Image, coords: Tuple[int]) -> Image.Image:
    roi = img.crop(coords).convert('L')
    enhancer = ImageEnhance.Contrast(roi)
    roi = enhancer.enhance(1.5)
    return roi

def extract_coords_from_bbox(bbox: BBox) -> Tuple[int]: return bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax

def extract_biggest_bbox(bboxes: List[BBox]) -> Union[None, BBox]:
    if len(bboxes) == 0:
        return None
    elif len(bboxes) == 1:
        return bboxes[0]
    else:
        sorted_bboxes_by_area = sorted([(bbox, bbox.area) for bbox in bboxes], key=lambda element: element[1])
        return sorted_bboxes_by_area[-1][0]

def read_text_from_roi(roi: Image.Image, textract) -> Union[None, Dict]:
    image_bytes = image_to_bts(np.array(roi))
    aws_response = textract.detect_document_text(Document={'Bytes': image_bytes})
    result = []
    for block in aws_response["Blocks"]:
        if block["BlockType"] == "LINE":
            bbox = block['Geometry']['BoundingBox']
            result.append({'text': block["Text"], 
                        'confidence': block["Confidence"]/100,
                        'bbox_width': bbox['Width'],
                        'bbox_height': bbox['Height'],
                        'bbox_area': bbox['Width']*bbox['Height']
                        })
    if len(result)>0: return sorted(result, key=lambda element: element['bbox_area'], reverse=True)
    else: return None