# Base image with matching Python (3.9) + CUDA 11.7 + development tools for compiling CUDA extensions
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

# System dependencies (minimal, focused on PEM + pointnet2 build + OpenGL for pyrender/blenderproc)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc g++ \
    git \
    wget curl ca-certificates \
    python3-dev \
    cmake \
    ninja-build \
    ffmpeg \
    libsm6 libxext6 \
    libgl1-mesa-dev libglu1-mesa-dev freeglut3-dev \
    libglib2.0-0 \
    unzip \
 && rm -rf /var/lib/apt/lists/*

# Upgrade pip tooling
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch (explicit versions as in environment.yaml) from official CUDA 11.7 wheels
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu117 \
    torch==2.0.0+cu117 torchvision==0.15.1+cu117

# Core PEM / point matching dependencies (mirrors environment.yaml & dependencies.sh; include ISM segmentation model deps)
# pytorch_lightning and ultralytics not needed for PEM, but useful for other tasks
RUN pip install --no-cache-dir \
    fvcore iopath \
    xformers==0.0.18 \
    torchmetrics==0.10.3 \
    blenderproc==2.6.1 \
    omegaconf ruamel.yaml hydra-colorlog hydra-core \
    gdown pandas imageio \
    pyrender pycocotools distinctipy \
    pytorch-lightning==1.8.6 \ 
    ultralytics==8.0.135 \ 
    timm gorilla-core==0.2.7.8 \
    trimesh==4.0.8 \
    gpustat==1.0.0 \
    imgaug einops \
    opencv-python \
    ninja 

# Install git-based packages
RUN pip install --no-cache-dir \
    git+https://github.com/facebookresearch/segment-anything.git

# Create workspace directory
#WORKDIR /workspace

# Copy only the Pose Estimation Model (PEM) code for precompilation
#COPY Pose_Estimation_Model ./Pose_Estimation_Model

# Build pointnet2 CUDA extension
#RUN cd Pose_Estimation_Model/model/pointnet2 && python setup.py install && \
#    rm -rf build

# Download pretrained PEM checkpoint
#RUN cd Pose_Estimation_Model && python download_sam6d-pem.py || echo "Checkpoint download step failed or already present."

###
# Install ros
# Install lsb-release first
RUN apt-get update && apt-get install -y lsb-release && rm -rf /var/lib/apt/lists/*
# add the keys
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -

RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ros-noetic-ros-base \
    ros-noetic-catkin \
    ros-noetic-vision-msgs \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

SHELL ["/bin/bash", "-c"]
RUN source /opt/ros/noetic/setup.bash
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
RUN source ~/.bashrc

# install python dependencies
RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3-rosdep \
    python3-rosinstall \
    python3-rosinstall-generator \
    python3-wstool \
    build-essential \
    python3-rosdep \
    python3-catkin-tools \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Initialize rosdep
RUN rosdep init
RUN rosdep update

# Create catkin workspace
RUN mkdir -p /root/catkin_ws/src
RUN /bin/bash -c  '. /opt/ros/noetic/setup.bash; cd /root/catkin_ws; catkin config -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPYTHON_INCLUDE_DIR=/usr/include/python3.8m -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.8.so; catkin build'

# clone and build message and service definitions
RUN /bin/bash -c 'cd /root/catkin_ws/src; \
                  git clone https://github.com/v4r-tuwien/object_detector_msgs.git'
RUN /bin/bash -c 'cd /root/catkin_ws/src; \
                  git clone https://gitlab.informatik.uni-bremen.de/robokudo/robokudo_msgs.git'
RUN /bin/bash -c '. /opt/ros/noetic/setup.bash; cd /root/catkin_ws; catkin build -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5'

RUN python3 -m pip install \
    catkin_pkg \
    rospkg

RUN python3 -m pip install \
    git+https://github.com/qboticslabs/ros_numpy.git

# Install mesa-utils for OpenGL support
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    mesa-utils \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*
# Set working directory for mounted code
WORKDIR /code

# (Optional) Set PYTHONPATH so scripts can run from anywhere
#ENV PYTHONPATH=/code/Pose_Estimation_Model:${PYTHONPATH}

# Default command (interactive shell); override with `docker run ... python train.py ...`
CMD ["/bin/bash", "-c", "source /opt/ros/noetic/setup.bash && source /root/catkin_ws/devel/setup.bash && exec bash"] 
    #&& \
    #python cnos_ros_wrapper.py dataset_name=ycbv model=cnos_fast model.onboarding_config.rendering_type=pyrender"]