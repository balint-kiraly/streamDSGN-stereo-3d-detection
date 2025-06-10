FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
LABEL authors="Balint Kiraly"

WORKDIR /app

# Install dependencies
RUN apt update && apt install -y \
    git \
    wget \
    python3-pip \
    python3-dev \
    g++

RUN pip install cython==0.29.36 \
    pycocotools==2.0.2 \
    cmake==3.31.0.1

# Pytorch
ENV USE_CUDA=1 \
    TORCH_CUDA_ARCH_LIST="8.9" \
    CMAKE_PREFIX_PATH="$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
ENV USE_CUDNN=0 \
    USE_FBGEMM=0

RUN git clone https://github.com/pytorch/pytorch && \
    cd pytorch && git checkout v1.7.1 && \
    git submodule sync && git submodule update --init --recursive && \
    pip install -r requirements.txt
RUN pip install ninja mkl-static mkl-include
WORKDIR /app/pytorch
RUN sed -i '/#include/ a #include <limits>' third_party/benchmark/src/benchmark_register.h
RUN sed -i '/#include <thrust\/functional.h>/ a #include <thrust/host_vector.h>' caffe2/utils/math_gpu.cu
RUN sed -i '/#include <vector>/ a #include <cstddef>' c10/util/hash.h
RUN python3 setup.py install # segmentation fault here

# COPY requirements.txt requirements.txt

# RUN pip install -r requirements.txt


# COPY . .


RUN echo build successful