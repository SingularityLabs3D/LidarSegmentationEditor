# FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel
FROM pavtiger/pytorch-notebook-cuda:cuda11.1.1-cudnn8-devel-ubuntu20.04-torch1.10.1-py3.8

USER root

# Fix GPG key issues
RUN apt update || true
RUN apt install -y curl 
RUN curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub | apt-key add -
RUN curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub | apt-key add -

RUN apt update && apt install -y git vim
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt .
RUN python -m pip install --upgrade pip
RUN pip install -U setuptools
RUN pip install -r requirements.txt

ENV TZ=America/New_York
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y wget libgl1-mesa-glx pcl-tools libc6 libstdc++6 build-essential cmake git libboost-all-dev


# RUN wget https://github.com/Kitware/CMake/releases/download/v3.17.2/cmake-3.17.2-Linux-x86_64.sh \
#       -q -O /tmp/cmake-install.sh \
#       && chmod u+x /tmp/cmake-install.sh \
#       && mkdir /usr/bin/cmake \
#       && /tmp/cmake-install.sh --skip-license --prefix=/usr/bin/cmake \
#       && rm /tmp/cmake-install.sh

# ENV PATH="/usr/bin/cmake/bin:${PATH}"

RUN pip install laspy
RUN pip install pybind11
RUN pip install opencv-python-headless

RUN pip install fastapi
RUN pip install uvicorn

COPY . /workspace/app
WORKDIR /workspace/app

RUN mkdir -p /workdir/lidar/LSK3DNet
RUN mkdir -p /workspace/lidar/LSK3DNet
RUN cp -r c_utils /workdir/lidar/LSK3DNet
RUN cp -r c_utils /workspace/lidar/LSK3DNet

RUN cd /workdir/lidar/LSK3DNet/c_utils \
&& mkdir -p build \
&& cd build \
&& cmake .. -Dpybind11_DIR=/opt/conda/lib/python3.8/site-packages/pybind11/share/cmake/pybind11 \
&& make

RUN export PYTHONPATH=/workdir/lidar/LSK3DNet/c_utils/build:$PYTHONPATH
ENV PYTHONPATH="/workdir/lidar/LSK3DNet/c_utils/build:${PYTHONPATH}"
ENV LD_LIBRARY_PATH="/workspace:${LD_LIBRARY_PATH}"

ENV LD_LIBRARY_PATH="/workspace/:$LD_LIBRARY_PATH"

RUN chown -R 1000:1000 /workspace/app
RUN chmod -R 755 /workspace/app
USER 1000:1000

ENTRYPOINT [ "python" , "test_skitti.py" ]
