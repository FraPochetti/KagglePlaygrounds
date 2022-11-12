import io, json
import mxnet as mx
from mxnet import image, gluon, nd
import numpy as np
from gluoncv.data.transforms.presets.imagenet import transform_eval

#ctx = mx.gpu()
ctx = mx.cpu()
print(f"ctx {ctx}")

def model_fn(model_dir):
    
    print(f"Model dir: {model_dir}")
    model = gluon.SymbolBlock.imports(
        model_dir + "/model-symbol.json",
        ["data"],
        model_dir + "/model-0000.params",
        ctx=ctx,
    )
    
    print("Loaded model-symbol model")
    print(f"Model type {type(model)}")
        
    labels = open(model_dir + "/code/classes_imagenet.txt", 'r')
    labels = [c.strip() for c in labels.readlines()]
    model.labels = labels
    print(f"Len labels: {len(labels)}; first 3 labels: {labels[:3]}")
          
    return model


def transform_fn(model, payload, input_content_type, output_content_type):
    
    io_bytes_obj = io.BytesIO(payload)
    print(f"Payload type {type(io_bytes_obj)}")
    
    npy_payload = np.load(io_bytes_obj)
    print(f"Converted payload to {type(npy_payload)} of shape {npy_payload.shape}")
    
    img = mx.nd.array(npy_payload)
    print(f"Converted numpy to {type(img)} of shape {img.shape}")
    
    img = img.as_in_context(ctx)
    img = transform_eval(img)
    print(f"Transformed mx array to {type(img)} of shape {img.shape}")
    
    output = model(img)
    pred = mx.nd.array(output[0])[None]
    print(f'Model prediction shape: {pred.shape}')
    
    topK = 5
    ind = nd.topk(pred, k=topK)[0].astype('int')
    
    result = {model.labels[ind[i].asscalar()]: str(nd.softmax(pred)[0][ind[i]].asscalar()) for i in range(topK)}
    print(f"Top 5 preds: {result}")
          
    return json.dumps(result)