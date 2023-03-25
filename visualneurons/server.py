import json
import sys
import textwrap

from tornado.ioloop import IOLoop
from tornado.websocket import WebSocketHandler

from .controller import StyleTransferController
from .serialization import base64_to_image


DEFAULT_ITERATIONS = 10


class StyleTransferSocket(WebSocketHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._controller = StyleTransferController(self)

    def check_origin(self, origin):
        # This method is used to allow/disallow connection from other servers
        return True

    async def open(self):
        # We load the model once the connection is open
        # an alternative would be to make the websocket load the model later on.
        print("Connection open")
        await self._controller.load_model()

    async def on_message(self, message):
        print(textwrap.shorten("Received message: {}".format(message), width=100))

        message = json.loads(message)

        if message["action"] == "close":
            self.close()
        if message["action"] == "request_image":
            data = message["data"]
            content_img = base64_to_image(data["content_image"])
            style_img = base64_to_image(data["style_image"])
            iterations = data.get("iterations", DEFAULT_ITERATIONS)
            await self._controller.request_image(
                content_img, style_img, num_iterations=iterations
            )
            self.close()
        else:
            raise Exception("invalid action")

    def on_close(self):
        print("Closing connection and stop the server")
        loop = IOLoop.current()
        loop.stop()

