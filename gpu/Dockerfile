# CUDA toolkit + compiler (nvcc) included
FROM nvidia/cuda:12.6.0-devel-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ca-certificates \
    build-essential cmake ninja-build \
    python3 python3-dev python3-pip python3-venv \
    libtbb-dev libembree-dev \
    libvtk9-dev \
 && rm -rf /var/lib/apt/lists/*

# Create a virtual environment and install build tools
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install -U pip build scikit-build-core pybind11

# Get ViennaLS sources
WORKDIR /src
RUN git clone https://github.com/ViennaTools/ViennaLS.git
WORKDIR /src/ViennaLS

# Install ViennaLS
RUN pip install .

# Get ViennaPS sources
WORKDIR /src
RUN git clone https://github.com/ViennaTools/ViennaPS.git
WORKDIR /src/ViennaPS

# Build with GPU enabled
ENV CMAKE_ARGS="-DVIENNAPS_USE_GPU=ON"

# Install ViennaPS
RUN pip install .
