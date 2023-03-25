import sys
import icevision
from icevision.all import *
from PIL import Image
import json, base64
from io import BytesIO

def format_response(message, status_code):
    return {
        "statusCode": str(status_code),
        "body": json.dumps(message),
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
        },
    }

def _get_size_without_padding(
    tfms_list, before_tfm_img, after_tfm_img
) -> Tuple[int, int]:
    height, width, _ = after_tfm_img.shape

    if get_transform(tfms_list, "Pad") is not None:
        after_pad_h, after_pad_w, _ = before_tfm_img.shape

        t = get_transform(tfms_list, "SmallestMaxSize")
        if t is not None:
            presize = t.max_size
            height, width = _func_max_size(after_pad_h, after_pad_w, presize, min)

        t = get_transform(tfms_list, "LongestMaxSize")
        if t is not None:
            size = t.max_size
            height, width = _func_max_size(after_pad_h, after_pad_w, size, max)

    return height, width


def py3round(number):
    """Unified rounding in all python versions."""
    if abs(round(number) - number) == 0.5:
        return int(2.0 * round(number / 2.0))

    return int(round(number))


def _func_max_size(height, width, max_size, func):
    scale = max_size / float(func(width, height))

    if scale != 1.0:
        height, width = tuple(py3round(dim * scale) for dim in (height, width))
    return height, width

def get_transform(tfms_list, t):
    for el in tfms_list:
        if t in str(type(el)):
            return el
    return None

labels = pickle.load(open("./model_dir/labels.pkl", "rb"))
class_map = ClassMap(labels)

infer_model = faster_rcnn.model(num_classes=len(class_map), pretrained=False)
infer_model.load_state_dict(torch.load('./model_dir/model_final.pth', map_location=torch.device('cpu')))

valid_tfms = tfms.A.Adapter([*tfms.A.resize_and_pad(896), tfms.A.Normalize()])

def handler(event, context): 
    body = json.loads(event['body'])
    image = base64.b64decode(body['data'])
    image = Image.open(BytesIO(image)).convert('RGB')

    # image = Image.open("./model_dir/test1.jpg") # TEST IMAGE
    print(f"Original image size: {image.size}")
    image = [np.asarray(image)]
    infer_ds = Dataset.from_images(images=image, tfm=valid_tfms)
    infer_dl = faster_rcnn.infer_dl(infer_ds, batch_size=1)
    x, sample = first(infer_dl)
    print(f"Size after pre-processing :{x[0].shape}")

    samples, preds = faster_rcnn.predict_dl(infer_model, infer_dl)

    img = draw_pred(img=samples[0]["img"], pred=preds[0], class_map=class_map, denormalize_fn=denormalize_imagenet)
    img = np.uint8(img)

    img_h, img_w = _get_size_without_padding([*tfms.A.resize_and_pad(896), tfms.A.Normalize()], image[0], img)

    p = int(min(img_h,img_w)*0.1)
    if img_h < img_w:
        pad = (img_w-img_h)//2
        img = img[(pad-p):(img_h+pad+p),:,:]
    else:
        pad = (img_h-img_w)//2
        img = img[:,(pad-p):(img_w+pad+p),:]
    
    img = Image.fromarray(img)
    fd = BytesIO()
    img.save(fd, format="PNG")

    return format_response(base64.b64encode(fd.getvalue()).decode(), 200)