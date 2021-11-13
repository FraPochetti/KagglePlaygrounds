FROM pytorch/pytorch:1.8.1-cuda10.2-cudnn7-runtime

RUN apt-get -y update && apt-get install -y gcc

RUN pip install pytorchvideo==0.1.3
RUN pip install pytorch-lightning==1.4.8
RUN pip install loguru
RUN pip install wandb
RUN pip3 install sagemaker-training

COPY ./code /opt/ml/code

# define train.py as the script entry point
ENV SAGEMAKER_PROGRAM train.py