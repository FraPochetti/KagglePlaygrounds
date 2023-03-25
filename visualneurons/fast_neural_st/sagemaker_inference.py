import os
import sys
import torch
from collections import defaultdict
import PIL
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models
from pathlib import Path
from random import shuffle
import torch.nn as nn
import torch.optim as optim
import io
import logging
import json
import base64

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
size = 300
padding = 30
JSON_CONTENT_TYPE = 'application/json'
JPEG_CONTENT_TYPE = 'image/jpeg'

#################################################
# NETWORK
#################################################

class TransformerNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Initial convolution layers
        self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(32, affine=True)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in2 = torch.nn.InstanceNorm2d(64, affine=True)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.in3 = torch.nn.InstanceNorm2d(128, affine=True)
        # Residual layers
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)
        # Upsampling Layers
        self.deconv1 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
        self.in4 = torch.nn.InstanceNorm2d(64, affine=True)
        self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.in5 = torch.nn.InstanceNorm2d(32, affine=True)
        self.deconv3 = ConvLayer(32, 3, kernel_size=9, stride=1)
        # Non-linearities
        self.relu = torch.nn.ReLU()

    def forward(self, X):
        y = self.relu(self.in1(self.conv1(X)))
        y = self.relu(self.in2(self.conv2(y)))
        y = self.relu(self.in3(self.conv3(y)))
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)
        y = self.relu(self.in4(self.deconv1(y)))
        y = self.relu(self.in5(self.deconv2(y)))
        y = self.deconv3(y)
        return y

class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out

class ResidualBlock(torch.nn.Module):
    """ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
    """

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        return out

class UpsampleConvLayer(torch.nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = torch.nn.functional.interpolate(x_in, mode='nearest', scale_factor=self.upsample)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out

#################################################
# IMAGE PROCESSING
#################################################

def compose(x, funcs, *args, order_key='_order', **kwargs):
    key = lambda o: getattr(o, order_key, 0)
    for f in sorted(list(funcs), key=key): x = f(x, **kwargs)
    return x

class Transform(): _order=0
        
class MakeRGB(Transform):
    def __call__(self, item): return {k: v.convert('RGB') for k, v in item.items()}

class ResizeFixed(Transform):
    _order=10
    def __init__(self, size):
        if isinstance(size,int): size=(size,size)
        self.size = size
        
    def __call__(self, item): return {k: v.resize(self.size, PIL.Image.BILINEAR) for k, v in item.items()}

class ToByteTensor(Transform):
    _order=20
    def to_byte_tensor(self, item):
        res = torch.ByteTensor(torch.ByteStorage.from_buffer(item.tobytes()))
        w,h = item.size
        return res.view(h,w,-1).permute(2,0,1)
    
    def __call__(self, item): return {k: self.to_byte_tensor(v) for k, v in item.items()}


class ToFloatTensor(Transform):
    _order=30
    def to_float_tensor(self, item): return item.float().div_(255.)
    
    def __call__(self, item): return {k: self.to_float_tensor(v) for k, v in item.items()}
    
class Normalize(Transform):
    _order=40
    def __init__(self, stats, p=None):
        self.mean = torch.as_tensor(stats[0] , dtype=torch.float32)
        self.std = torch.as_tensor(stats[1] , dtype=torch.float32)
        self.p = p
    
    def normalize(self, item): return item.sub_(self.mean[:, None, None]).div_(self.std[:, None, None])
    def pad(self, item): return nn.functional.pad(item[None], pad=(self.p,self.p,self.p,self.p), mode='replicate').squeeze(0)
    
    def __call__(self, item): 
        if self.p is not None: return {k: self.pad(self.normalize(v)) for k, v in item.items()}
        else: return {k: self.normalize(v) for k, v in item.items()}

class DeProcess(Transform):
    _order=50
    def __init__(self, stats, size=None, p=None, ori=None):
        self.mean = torch.as_tensor(stats[0] , dtype=torch.float32)
        self.std = torch.as_tensor(stats[1] , dtype=torch.float32)
        self.size = size
        self.p = p
        self.ori = ori
    
    def de_normalize(self, item): return ((item*self.std[:, None, None]+self.mean[:, None, None])*255.).clamp(0, 255)
    def rearrange_axis(self, item): return np.moveaxis(item, 0, -1)
    def to_np(self, item): return np.uint8(np.array(item))
    def crop(self, item): return item[self.p:self.p+self.size,self.p:self.p+self.size,:]
    def de_process(self, item): 
        return PIL.Image.fromarray(self.crop(self.rearrange_axis(self.to_np(self.de_normalize(item))))).resize(self.ori, PIL.Image.BICUBIC)
                
    def __call__(self, item): 
        if isinstance(item, torch.Tensor): return self.de_process(item) 
        if isinstance(item, tuple): return tuple([self.de_process(v) for v in item])
        if isinstance(item, dict): return {k: self.de_process(v) for k, v in item.items()}

#################################################
# SAGEMAKER INFERENCE FUNCTIONS
#################################################

def image_to_base64(image):
    # Make the image the correct format
    fd = io.BytesIO()
    # Save the image as PNG
    image.save(fd, format="PNG")
    return base64.b64encode(fd.getvalue())


def base64_to_image(data: bytes) -> np.ndarray:
    """Convert an image in base64 to a numpy array"""
    b64_image = base64.b64decode(data)
    fd = io.BytesIO(b64_image)
    img = PIL.Image.open(fd)
    #img_data = np.array(img).astype("float32")

    #if img_data.shape[-1] == 4:
    #    # We only support rgb
    #    img_data = img_data[:, :, :3]

    return img


def model_fn(model_dir):
    logger.info('model_fn')
    device = torch.device("cpu")
    model = TransformerNet()
    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f, map_location=device))
    return model.to(device)

def input_fn(request_body, content_type=JPEG_CONTENT_TYPE):
    img = PIL.Image.open(io.BytesIO(request_body))
    item = {'input': img}

    rgb = MakeRGB()
    resized = ResizeFixed(size)
    tobyte = ToByteTensor()
    tofloat = ToFloatTensor()
    norm = Normalize(imagenet_stats, padding)
    tmfs = [rgb, resized, tobyte, tofloat, norm]
    item = compose(item, tmfs)
    
    return {'img': item['input'], 'size': img.size}

def predict_fn(input_object, model):
    img = input_object['img']
    device = torch.device("cpu")
    out = model(img[None].to(device))
    input_object['img'] = out[0].detach() 
    return input_object

def output_fn(prediction, content_type=JSON_CONTENT_TYPE):
    p = prediction['img']
    original_size = prediction['size']
    denorm = DeProcess(imagenet_stats, size, padding, original_size)
    pred = denorm(p)
    if content_type == JSON_CONTENT_TYPE: return json.dumps({'prediction': image_to_base64(pred).decode()})
    