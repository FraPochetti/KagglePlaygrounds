FROM python:3.8

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install torch==1.10.0+cu111 torchvision==0.11.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html --upgrade
RUN pip install icevision[all]==0.12.0 -U
RUN pip install yolov5-icevision -U
RUN pip install awscli -U
RUN pip install boto3==1.21.32 -U