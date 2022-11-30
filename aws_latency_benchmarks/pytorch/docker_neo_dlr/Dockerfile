FROM nvcr.io/nvidia/tensorrt:19.12-py3

ENV DEBIAN_FRONTEND=noninteractive 
RUN apt-get update
RUN pip install --upgrade pip
RUN pip install boto3
RUN pip install sagemaker
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu102

RUN git clone --recursive https://github.com/neo-ai/neo-ai-dlr && \
    cd neo-ai-dlr && \
    mkdir build && \
    cd build && \
    cmake .. -DUSE_CUDA=ON -DUSE_CUDNN=ON -DUSE_TENSORRT=/opt/tensorrt && \
    make -j4 && \
    cd ../python && \
    python3 setup.py install --user