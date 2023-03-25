"""Serialization utilities to communicate between client and server."""
import base64
import numpy as np

from io import BytesIO
from PIL import Image


def image_to_base64(image: np.ndarray) -> bytes:
    """Convert a numpy array to a png image encoded in base64.

    Numpy array is a RGB image of shape (width, height, 3) where each
    value ranges between 0 and 255.  
    """
    # Make the image the correct format
    image = image.astype("uint8")
    pil_image = Image.fromarray(image)
    fd = BytesIO()
    # Save the image as PNG
    pil_image.save(fd, format="PNG")
    return base64.b64encode(fd.getvalue())


def base64_to_image(data: bytes) -> np.ndarray:
    """Convert an image in base64 to a numpy array"""
    b64_image = base64.b64decode(data)
    fd = BytesIO(b64_image)
    img = Image.open(fd)
    img_data = np.array(img).astype("float32")

    if img_data.shape[-1] == 4:
        # We only support rgb
        img_data = img_data[:, :, :3]

    return img_data
