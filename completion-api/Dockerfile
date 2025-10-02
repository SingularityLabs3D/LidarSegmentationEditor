FROM alexsilener/ros_noetic_minkowski

RUN git clone https://github.com/astra-vision/LiDPM.git; \
    mv LiDPM/lidpm lidpm

RUN git clone https://github.com/PRBonn/LiDiff.git; \
    mv LiDiff/lidiff lidiff

RUN wget https://github.com/astra-vision/LiDPM/releases/download/v1.0-weights/lidpm_diffusion.ckpt; \
    wget https://www.ipb.uni-bonn.de/html/projects/lidiff/refine_net.ckpt

RUN pip install \
    open3d==0.13 \
    pykeops==2.1.2 \
    ninja \
    tqdm \
    click \
    pytorch_lightning==1.7 \
    six \
    hdbscan \
    natsort \
    diffusers \
    tensorboard \
    torchmetrics==0.7.0 \
    imageio \
    fpsample

RUN python3 -m pip install --upgrade pip
RUN pip install -U setuptools
RUN pip install open3d==0.17
RUN pip uninstall importlib-metadata -y
RUN pip install importlib-metadata
RUN pip install jupyterlab

RUN pip install fastapi uvicorn python-multipart python-dotenv
COPY . .
RUN mkdir "exp"
ENTRYPOINT [ "uvicorn", "--host", "0.0.0.0", "main:app" ] 
