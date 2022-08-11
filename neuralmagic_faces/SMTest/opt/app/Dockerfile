FROM ubuntu:18.04

RUN apt-get -y update
RUN apt-get install -y python3.6
RUN apt-get install -y --no-install-recommends python3-pip
RUN apt-get install -y python3-setuptools

RUN ln -s /usr/bin/python3 /usr/bin/python & \
    ln -s /usr/bin/pip3 /usr/bin/pip
    
RUN pip install flask
RUN pip install pandas
RUN pip install scikit-learn
RUN pip install joblib


WORKDIR /usr/local/bin
COPY serve /usr/local/bin/serve

EXPOSE 8080