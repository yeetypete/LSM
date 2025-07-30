FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID
ARG TORCH_CUDA_ARCH_LIST="12.0+PTX"

# hadolint ignore=DL3008
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update && apt-get install --no-install-recommends -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && rm -rf /var/lib/apt/lists/*

# hadolint ignore=DL3008
RUN apt-get update && apt-get install --no-install-recommends -y \
    curl \
    git \
    git-lfs \
    libgl1 \
    libglib2.0-0 \
    nano \
    python-is-python3 \
    python3-dev \
    python3-pip \
    ssh \
    tmux \
    vim \
    wget \
    && rm -rf /var/lib/apt/lists/*

# hadolint ignore=SC2174
RUN mkdir -p -m 0600 ~/.ssh && ssh-keyscan github.com >> ~/.ssh/known_hosts

# hadolint ignore=DL3013,DL3042
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=requirements.txt \
    --mount=type=ssh \
    export TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST && \
    pip install --index-url=https://download.pytorch.org/whl/cu128 \
    torch \
    torchvision && \
    pip install -r requirements.txt && \
    pip install flash-attn --no-build-isolation && \
    pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.7.0+cu128.html

# Copy submodules and install all components
RUN --mount=type=bind,source=.,target=/workspace \
    cp -r /workspace/submodules /tmp/ && \
    export TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST && \
    pip install /tmp/submodules/PointTransformerV3/Pointcept/libs/pointops && \
    pip install /tmp/submodules/3d_gaussian_splatting/diff-gaussian-rasterization && \
    pip install /tmp/submodules/3d_gaussian_splatting/simple-knn && \
    rm -rf /tmp/submodules

# create cache directory
RUN mkdir -p /home/$USERNAME/.cache && chown -R $USERNAME:$USERNAME /home/$USERNAME/.cache

USER $USERNAME
