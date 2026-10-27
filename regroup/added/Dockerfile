# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ── System packages ────────────────────────────────────────────────────────────
# apt cache is mounted so debs aren't re-fetched across builds
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        curl gnupg lsb-release ca-certificates \
        python3-pip python3-dev \
        libgl1 libglib2.0-0

# ── ROS 2 Humble ───────────────────────────────────────────────────────────────
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
        -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
        http://packages.ros.org/ros2/ubuntu jammy main" \
        > /etc/apt/sources.list.d/ros2.list \
    && apt-get update && apt-get install -y --no-install-recommends \
        ros-humble-ros-base \
        ros-humble-cv-bridge \
        ros-humble-vision-msgs \
        ros-humble-rosbag2-py \
        ros-humble-rosbag2-storage-default-plugins \
        ros-humble-rosbag2-storage-mcap \
        ffmpeg

# ── Python: PyTorch (large — cached so layer rebuilds don't re-download) ───────
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install \
    torch==2.3.1+cu121 torchvision==0.18.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# ── Python: ML + ROS support ───────────────────────────────────────────────────
# numpy<2 is required: ros-humble-cv-bridge is compiled against the NumPy 1.x
# ABI and raises "_ARRAY_API not found" at import time with NumPy 2.x.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install \
    "numpy<2" \
    ultralytics \
    "boxmot==19.0.0" \
    scipy \
    scikit-learn \
    Pillow

# ── REGROUP package ────────────────────────────────────────────────────────────
WORKDIR /opt/regroup
COPY regroup/ .

# Pre-download OSNet ReID weights into boxmot's internal weights dir.
# Done at build time so the container starts without needing internet access.
RUN python3 -c "from boxmot.reid.core.reid import ReID; ReID(device='cpu')" \
    && echo "ReID weights downloaded."

# ── Entrypoint ─────────────────────────────────────────────────────────────────
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
# CMD ["regroup_detector"]
