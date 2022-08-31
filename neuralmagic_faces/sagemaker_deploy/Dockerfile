FROM python:3.9

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install opencv-python-headless==4.6.0.66
RUN pip install deepsparse[yolo]==1.0.2   
RUN pip install flask

WORKDIR /usr/local/bin
COPY serve /usr/local/bin/serve

EXPOSE 8080