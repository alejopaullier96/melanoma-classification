# FROM ubuntu:18.04
FROM nvidia/cuda:10.2-cudnn7-runtime-ubuntu18.04

# Set the Conda path in the path variables
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

# Install htop, Python3, wget
RUN apt update \
    && apt install -y htop python3-dev wget

# Install MiniConda for Linux
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir root/.conda \
    && sh Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

# Create virtual environment
RUN conda create -y -n my_venv python=3.9

# Copy all files in current working directory to src/
COPY . src/

# Activate virtual environment and install requirements
RUN /bin/bash -c "cd src \
    && source activate my_venv \
    && pip install -r requirements.txt"
