FROM python:3.9

RUN apt-get update

RUN pip install deepsparse-nightly   
RUN pip install flask

ENV NM_FAST_VNNI_EMULATION=1 
ENV NM_BIND_THREADS_TO_CORES=1

WORKDIR /usr/local/bin
COPY serve /usr/local/bin/serve

EXPOSE 8080