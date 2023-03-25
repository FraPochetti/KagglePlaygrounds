"""Websocket client for testing"""
import json
import os
import textwrap

import numpy as np
from PIL import Image

from tornado.websocket import websocket_connect
from tornado.ioloop import IOLoop

from .images import load_image
from .serialization import image_to_base64, base64_to_image
from .controller import State

OUTPUT_DIR = "./output"
SAMPLE_STYLE_IMAGE = "prototypes/styletransfer/tests/MonetLookingForward.jpg"
SAMPLE_CONTENT_IMAGE = "prototypes/styletransfer/tests/img_lights.jpg"


class WebsocketClient:
    """Websocket client for testing our style transfer server"""

    def __init__(self, url: str):
        self.url = url

    async def connect(self):
        self.ws = await websocket_connect(self.url)

        while True:
            msg = await self.ws.read_message()
            print(
                textwrap.shorten("Message received from server:\n#{}#".format(msg), 100)
            )
            if msg is None:
                # Con nection is closed
                print("Connection is closed")
                break

            msg = json.loads(msg)

            if msg["state"] == State.MODEL_LOADED.value:
                content_img = load_image(SAMPLE_CONTENT_IMAGE)
                style_img = load_image(SAMPLE_STYLE_IMAGE)
                msg = _request_image_message(content_img, style_img)
                await self.ws.write_message(msg)
            elif msg["state"] == State.END_ITERATION.value:
                # The model has completed an iteration and we handle it
                _handle_end_iteration(msg)
            elif msg["state"] == State.END.value:
                loop = IOLoop.current()
                loop.stop()
            else:
                print("Received an unhandled message {}".format(msg["state"]))


def _handle_end_iteration(msg):
    image = Image.fromarray(base64_to_image(msg["data"]["image"]).astype("uint8"))

    iteration = msg["data"]["iteration"]

    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    with open(
        os.path.join(OUTPUT_DIR, "image_{:04d}.png".format(iteration)), "wb"
    ) as fd:
        image.save(fd, format="PNG")


def _request_image_message(content_img: np.ndarray, style_img: np.ndarray):
    return json.dumps(
        {
            "action": "request_image",
            "data": {
                "content_image": image_to_base64(content_img).decode(),
                "style_image": image_to_base64(style_img).decode(),
            },
        }
    )


if __name__ == "__main__":
    loop = IOLoop.instance()
    client = WebsocketClient("ws://localhost:8000/styletransfer")
    loop.spawn_callback(client.connect)
    loop.start()
