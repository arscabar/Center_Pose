# Use a NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Set environment variables for non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    libgl1-mesa-glx \
    libegl1-mesa \
    libxkbcommon-x11-0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-render-util0 \
    libxcb-xinerama0 \
    libxcb-xfixes0 \
    libfontconfig1 \
    libharfbuzz0b \
    libfreetype6 \
    libpng16-16 \
    libtiff5 \
    libjpeg-turbo8 \
    libwebp7 \
    libosmesa6-dev \
    libxrender1 \
    libxext6 \
    libsm6 \
    libxcb-cursor0 \
    libxcb-util1 \
    libx11-xcb1 \
    libxcb-xkb1 \
    libxcb-render0 \
    libxcb-shm0 \
    libglib2.0-0 \
    libxcb-shape0 \
    libxcb-randr0 \
    libxcb-xinput0 \
    libxcb-sync1 \
    libdbus-1-3 \
    libglu1-mesa-dev \
    freeglut3-dev \
    mesa-common-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Miniforge3
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O ~/miniforge.sh && \
    /bin/bash ~/miniforge.sh -b -p /opt/conda && \
    rm ~/miniforge.sh
ENV PATH=$CONDA_DIR/bin:$PATH

# Create conda environment
RUN conda create -n 4D-humans python=3.10 -y

# Set environment variables
ENV CONDA_DEFAULT_ENV=4D-humans
ENV PATH=/opt/conda/envs/4D-humans/bin:$PATH

# 2. Install PyTorch with CUDA 12.1 support
# Use strict channel priority to ensure pytorch channel is preferred over conda-forge
RUN /bin/bash -c "source activate 4D-humans && \
    conda config --env --set channel_priority strict && \
    conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -c conda-forge && \
    echo '=== Verifying PyTorch CUDA installation ===' && \
    python -c 'import torch; print(f\"PyTorch version: {torch.__version__}\"); print(f\"CUDA available: {torch.cuda.is_available()}\"); print(f\"CUDA version: {torch.version.cuda}\"); assert torch.cuda.is_available(), \"ERROR: CUDA is not available in PyTorch!\"' && \
    echo '=== PyTorch CUDA verification successful! ==='"

# 3. Install Detectron2
RUN /bin/bash -c "source activate 4D-humans && \
    pip install --no-build-isolation git+https://github.com/facebookresearch/detectron2.git"

# 4. Install remaining dependencies manually
RUN /bin/bash -c "source activate 4D-humans && \
    pip uninstall -y opencv-python && \
    pip install \
    pyopengl==3.1.4 \
    opencv-python-headless \
    scikit-image \
    scipy \
    tqdm \
    yacs \
    termcolor \
    fvcore \
    iopath \
    cloudpickle \
    omegaconf \
    pyrender \
    trimesh \
    smplx \
    ultralytics \
    lap \
    PySide6 \
    pytorch-lightning \
    timm \
    pandas \
    open3d \
    gradio \
    && pip cache purge"

# 5. Install chumpy separately with no-build-isolation to avoid setup.py issues
RUN /bin/bash -c "source activate 4D-humans && \
    pip install --no-build-isolation chumpy"

# Set Python environment variables for pyrender
ENV PYOPENGL_PLATFORM=osmesa

# Copy the entire project into the container
WORKDIR /app
COPY . /app

# Install additional dependencies for 4D-Humans
RUN /bin/bash -c "source activate 4D-humans && \
    pip install gdown einops webdataset dill"

# Clone and manually install 4D-Humans to avoid chumpy build issues
RUN /bin/bash -c "source activate 4D-humans && \
    cd /tmp && \
    git clone https://github.com/shubham-goel/4D-Humans.git && \
    cd 4D-Humans && \
    pip install --no-deps -e ."

# [Mounting Point] Create cache directory for binding
RUN mkdir -p /root/.cache/4DHumans

# Set Qt platform to XCB for X11 forwarding
ENV QT_QPA_PLATFORM=xcb

# Command to run the application
CMD ["/bin/bash", "-c", "source activate 4D-humans && python main_4d_hybrid.py"]