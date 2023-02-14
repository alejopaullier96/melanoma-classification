FROM nvidia/cuda:10.2-cudnn7-runtime-ubuntu18.04

# These command is set because python3-opencv prompts the user a couple questions which can be ignored.
ENV DEBIAN_FRONTEND=noninteractive
# This command is due to a recurrent NVIDIA cuda / Docker GPG error
# Please check: https://lifesaver.codes/answer/cuda-docker-gpg-error-619
RUN rm /etc/apt/sources.list.d/cuda.list

# Set the Conda path in the path variables
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

# Install htop, Python3, wget
RUN apt update \
    && apt install -y htop python3-dev wget \
    && apt install -y python3-opencv \
    && apt install -y nginx

# Create symbolic links
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN ln -s /usr/bin/pip3 /usr/bin/pip

# Copy all files in current working directory to opt/program
COPY ./container opt/program

# Set some environment variables. PYTHONUNBUFFERED keeps Python from buffering our standard
# output stream, which means that logs can be delivered to the user quickly. PYTHONDONTWRITEBYTECODE
# keeps Python from writing the .pyc files which are unnecessary in this case. We also update
# PATH so that the train and serve programs are found when the container is invoked.
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

# Install MiniConda for Linux
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

# Create virtual environment
RUN conda create -y -n my_venv python=3.9

# Install requirements
RUN /bin/bash -c "cd opt/program \
    && source activate my_venv \
    && pip3 install -r requirements.txt"

# Set working directory
WORKDIR /opt/program

RUN ["chmod", "+x", "/opt/program/entrypoint.sh"]

EXPOSE 5000

# Activate the conda environment at runtime
ENTRYPOINT ["./entrypoint.sh"]