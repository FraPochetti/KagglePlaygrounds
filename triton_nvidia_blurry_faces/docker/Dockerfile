FROM nvcr.io/nvidia/tensorrt:21.08-py3

ENV DEBIAN_FRONTEND=noninteractive 
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get install -y python3.8-tk
RUN pip install --force pip~=21.0
RUN pip install torch==1.10.0+cu111 torchvision==0.11.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html -U
RUN pip install icevision[all]==0.12.0 -U
RUN pip install icedata==0.5.1 -U

RUN pip install awscli -U 
RUN pip install boto3==1.21.32 -U
RUN pip install sagemaker==2.82.2 -U
RUN pip install nvidia-pyindex
RUN pip install tritonclient[http]==2.20.0
RUN pip install onnx==1.11.0
RUN pip install onnxruntime==1.11.0