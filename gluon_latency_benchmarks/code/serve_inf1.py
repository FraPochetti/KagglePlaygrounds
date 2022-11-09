import io, json, os
import mxnet as mx
from mxnet import image, gluon, nd
import numpy as np
from gluoncv.data.transforms.presets.imagenet import transform_eval
from collections import namedtuple

ctx = mx.neuron()
print(f"ctx {ctx}")

def model_fn(model_dir):
    Batch = namedtuple("Batch", ["data"])
    dtype = "float32"

    sym, arg_params, aux_params = mx.model.load_checkpoint(os.path.join(model_dir, "compiled"), 0)
    model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
    for arg in arg_params:
        arg_params[arg] = arg_params[arg].astype(dtype)

    for arg in aux_params:
        aux_params[arg] = aux_params[arg].astype(dtype)

    exe = model.bind(
        for_training=False, data_shapes=[("data", (1, 3, 224, 224))], label_shapes=model._label_shapes
    )
    model.set_params(arg_params, aux_params, allow_missing=True)
    
    labels = open(model_dir + "/code/classes_imagenet.txt", 'r')
    labels = [c.strip() for c in labels.readlines()]
    model.labels = labels
    print(f"Len labels: {len(labels)}; first 3 labels: {labels[:3]}")
          
    return model


def transform_fn(model, payload, input_content_type, output_content_type):
    
    Batch = namedtuple("Batch", ["data"])
    
    io_bytes_obj = io.BytesIO(payload)
    print(f"Payload type {type(io_bytes_obj)}")
    
    npy_payload = np.load(io_bytes_obj)
    print(f"Converted payload to {type(npy_payload)} of shape {npy_payload.shape}")
    
    img = mx.nd.array(npy_payload)
    print(f"Converted numpy to {type(img)} of shape {img.shape}")
    
    img = img.as_in_context(ctx)
    img = transform_eval(img)
    print(f"Transformed mx array to {type(img)} of shape {img.shape}")
    
    model.forward(Batch([img]))
    pred = model.get_outputs()[0].asnumpy()
    print(f'Model prediction shape: {pred.shape}', type(pred))
    
    result = np.squeeze(pred)
    result_exp = np.exp(result - np.max(result))
    result = result_exp / np.sum(result_exp)
    ind = np.argmax(result)
    result = {model.labels[ind]: result[ind]}
          
    return json.dumps(result)