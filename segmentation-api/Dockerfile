FROM docker pull pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel

COPY . .

USER root

RUN pip install -r requirements.txt

RUN cd src/map_building/c_utils \
    && mkdir -p build \
    && cd build \
    && cmake .. -Dpybind11_DIR=/usr/local/lib/python3.6/dist-packages/pybind11/share/cmake/pybind11 \
    && make

