FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    curl gnupg software-properties-common \
    git build-essential pkg-config python3 python3-venv python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Add Intel graphics PPA and required compute / media libraries
RUN add-apt-repository -y ppa:kobuk-team/intel-graphics \
    && apt-get update \
    && apt-get install -y \
    libze-intel-gpu1 libze1 libze-dev \
    intel-opencl-icd clinfo \
    intel-metrics-discovery intel-gsc \
    intel-ocloc \
    intel-media-va-driver-non-free libmfx-gen1 libvpl2 libvpl-tools libva-glx2 va-driver-all \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | bash
ENV PATH="/root/.local/bin:${PATH}"

RUN apt update && apt install -y pciutils unzip

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

WORKDIR /workspace
