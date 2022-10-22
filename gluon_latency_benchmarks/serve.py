import mxnet as mx
from mxnet import image, gluon
import gluoncv
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
print("I am here")


def model_fn(model_dir):
    logging.info("Invoking user-defined model_fn")
    ctx = mx.cpu(0)
    
    print("ctx {}".format(ctx))
    print(f"Model dir: {model_dir}"")
    model = gluon.SymbolBlock.imports(
        "model-symbol.json",
        ["data"],
        "model-0000.params",
        ctx=ctx,
    )
          
    return model


def transform_fn(model, payload, input_content_type, output_content_type):
    print("Inside transform_fn")

    logging.info("Invoking user-defined transform_fn")

    if input_content_type != "application/x-npy":
        raise RuntimeError("Input content type must be application/x-npy")

    io_bytes_obj = io.BytesIO(payload)
    npy_payload = np.load(io_bytes_obj)
    print(npy_payload.shape)

   # TO BE FILLED LATER ON AFTER FIGURING OUT THE BUG
          
    return {"result": "ok"}