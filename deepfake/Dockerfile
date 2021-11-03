FROM pytorch/pytorch:1.8.1-cuda10.2-cudnn7-runtime

RUN pip install pytorchvideo==0.1.3
RUN pip install pytorch-lightning==1.4.8
RUN pip install loguru
RUN pip install wandb

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

COPY ./code /opt/program
WORKDIR /opt/program