# Use a NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

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

# Copy environment files
COPY 4D-Humans_disabled/environment.yml /tmp/environment.yml
COPY 4D-Humans_disabled/hmr2 /4D-Humans_disabled/hmr2

# 1. Create environment WITHOUT pip packages first
RUN sed -i '/pip:/Q' /tmp/environment.yml && \
    conda env create -f /tmp/environment.yml

# Set environment variables
ENV CONDA_DEFAULT_ENV=4D-humans
ENV PATH=/opt/conda/envs/4D-humans/bin:$PATH

# 2. Install PyTorch, Numpy & Detectron2 explicitly
RUN /bin/bash -c "source activate 4D-humans && \
    conda install -y numpy pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia && \
    pip install --no-build-isolation git+https://github.com/facebookresearch/detectron2.git"

# 3. Install remaining dependencies manually
RUN /bin/bash -c "source activate 4D-humans && \
    pip uninstall -y opencv-python && \
    pip install --no-build-isolation \
    pyopengl==3.1.4 \
    opencv-python-headless \
    scikit-image \
    scipy \
    chumpy \
    tqdm \
    yacs \
    termcolor \
    fvcore \
    iopath \
    cloudpickle \
    omegaconf \
    pyrender --upgrade \
    trimesh \
    smplx \
    ultralytics \
    PySide6 \
    pytorch-lightning \
    timm \
    pandas \
    open3d \
    gradio \
    && pip cache purge"

# Set Python environment variables for pyrender
ENV PYOPENGL_PLATFORM=osmesa

# Copy the entire project into the container
WORKDIR /app
COPY . /app

# [FIX] Install 4D-Humans from local source (with our fixes) instead of git
RUN /bin/bash -c "source activate 4D-humans && \
    cd /app/4D-Humans_disabled && \
    pip install -e ."

# --- Apply code modifications inside Docker ---

# 1. Correct sys.path
RUN sed -i 's|sys.path.append(os.path.join(current_dir, "4D-Humans_disabled"))|sys.path.append(os.path.join("/app", "4D-Humans_disabled"))|g' /app/main_4d_hybrid.py
RUN sed -i 's|sys.path.append(os.path.join(current_dir, "4D-Humans_disabled"))|sys.path.append(os.path.join("/app", "4D-Humans_disabled"))|g' /app/web_ui.py

# 2. Set PYOPENGL_PLATFORM in renderer.py to 'osmesa'
RUN sed -i "s|os.environ\['PYOPENGL_PLATFORM'\] = 'egl'|os.environ['PYOPENGL_PLATFORM'] = 'osmesa'|g" /app/4D-Humans_disabled/hmr2/utils/renderer.py
RUN sed -i "s|os.environ\['PYOPENGL_PLATFORM'\] = 'opengl'|os.environ['PYOPENGL_PLATFORM'] = 'osmesa'|g" /app/4D-Humans_disabled/hmr2/utils/renderer.py

# 3. Add weights_only=False
RUN sed -i 's|model = HMR2.load_from_checkpoint(checkpoint_path, strict=False, cfg=model_cfg)|model = HMR2.load_from_checkpoint(checkpoint_path, strict=False, cfg=model_cfg, weights_only=False)|g' /app/4D-Humans_disabled/hmr2/models/__init__.py

# [CRITICAL FIX FINAL] 4. Use 'sed' to patch renderer.py (Safe & Fast)
# This replaces the line causing the crash with a conditional logic, without using python -c
RUN sed -i "s|image = cv2.imread(imgname).astype(np.float32)\[:, :, ::-1\] / 255.|image = (cv2.imread(imgname) if isinstance(imgname, str) else imgname).astype(np.float32)[:, :, ::-1] / 255.|g" /app/4D-Humans_disabled/hmr2/utils/renderer.py

# [Mounting Point] Create cache directory for binding
RUN mkdir -p /root/.cache/4DHumans

# Set Qt platform to XCB for X11 forwarding
ENV QT_QPA_PLATFORM=xcb

# Command to run the application
CMD ["/bin/bash", "-c", "source activate 4D-humans && python main_4d_hybrid.py"]