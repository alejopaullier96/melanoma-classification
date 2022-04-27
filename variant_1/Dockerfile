FROM nvidia/cuda:10.2-cudnn7-runtime-ubuntu18.04

# Install htop, Python3, wget
RUN apt update \
    && apt install -y htop python3-dev wget \
    && apt install -y python3-opencv

# Copy all files in current working directory to opt/program
COPY ./container opt/program

# Set some environment variables. PYTHONUNBUFFERED keeps Python from buffering our standard
# output stream, which means that logs can be delivered to the user quickly. PYTHONDONTWRITEBYTECODE
# keeps Python from writing the .pyc files which are unnecessary in this case. We also update
# PATH so that the train and serve programs are found when the container is invoked.
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

# Install requirements
RUN /bin/bash -c "cd opt/program \
    && pip3 install -r requirements.txt"

WORKDIR /opt/program
