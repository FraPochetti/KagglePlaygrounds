import enum

import numpy as np

from tornado.ioloop import IOLoop
from tornado.websocket import WebSocketHandler

from .async_utils import async_next, AsyncStopIteration
from .model import StyleTransfer, StyleTransferResult, make_google_style_transfer
from .serialization import image_to_base64


class InvalidTransition(Exception):
    pass


class State(enum.Enum):
    INIT = "init"
    MODEL_LOADING = "model_loading"
    MODEL_LOADED = "model_loaded"
    START_ITERATION = "start_iteration"
    END_ITERATION = "end_iteration"
    END = "end"


class StyleTransferController:
    def __init__(self, ws: WebSocketHandler):
        """Controls the interaction between the StyleTransfer and
        the external world, through a websocket handler.

        The controller can be in states specified by `State` and its
        methods correspond to its available actions.
        """
        self._ws = ws
        self.state = State.INIT
        self._model = None

    # Transitions
    async def load_model(self):
        if self.state != State.INIT:
            raise InvalidTransition()

        await self._set_and_send_state(State.MODEL_LOADING)

        # We run model initialization in a separate thread so that
        # we don't block the tornado event loop.
        loop = IOLoop.current()
        self._model: StyleTransfer = await loop.run_in_executor(
            None, self._do_load_model
        )

        await self._set_and_send_state(State.MODEL_LOADED)

    async def request_image(
        self, content_img: np.ndarray, style_img: np.ndarray, num_iterations=10
    ):
        if self.state != State.MODEL_LOADED:
            raise InvalidTransition()
        loop = IOLoop.current()

        # We can't run a for loop because we risk of blocking everything, therefore
        # we need to invoke the `next` method into a separate threads (executor).

        # TODO: this could be converted to an `async for` loop by creating an
        # asynchronous iterator. However, it may not be worth the time (just syntactinc sugar)
        iterator = self._model.run_style_transfer(
            content_img, style_img, num_iterations=num_iterations
        )

        while True:
            await self._set_and_send_state(State.START_ITERATION)

            try:
                style_results: StyleTransferResult = await loop.run_in_executor(
                    None, async_next, iterator
                )
            except AsyncStopIteration:
                await self._set_and_send_state(State.END)
                break

            await self._set_and_send_state(
                State.END_ITERATION,
                {
                    "iteration": style_results.iteration_no,
                    "image": image_to_base64(style_results.image).decode(),
                },
            )

    async def _set_and_send_state(self, state: State, data=None):
        self.state = state
        await self._ws.write_message({"state": self.state.value, "data": data})

    def _do_load_model(self) -> StyleTransfer:
        return make_google_style_transfer()
