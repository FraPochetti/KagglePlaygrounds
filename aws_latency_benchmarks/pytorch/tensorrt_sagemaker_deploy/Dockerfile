FROM nvcr.io/nvidia/pytorch:22.10-py3
RUN apt-get update

WORKDIR /usr/local/bin
COPY serve /usr/local/bin/serve

EXPOSE 8080