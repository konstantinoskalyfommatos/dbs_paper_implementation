FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

USER root

# Install Python and basic dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    python3-venv \
    curl \
    unzip \
    sudo \
    git \
    && rm -rf /var/lib/apt/lists/*

# Make Python 3.11 the default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --set python3 /usr/bin/python3.11 && \
    ln -sf /usr/bin/python3 /usr/bin/python

RUN pip install --upgrade pip
RUN pip install uv
RUN uv --version

# Set up workspace
WORKDIR /app
COPY requirements_retrieval.txt ./requirements_retrieval.txt

# Install Python dependencies
RUN uv pip install --system -r ./requirements_retrieval.txt


ENV HOME=/app
COPY . /app

CMD ["bash"]