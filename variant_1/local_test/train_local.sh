#!/bin/sh

# Get first positional argument and store as variable
image=$1

# Create directories.
mkdir -p test_dir/model
mkdir -p test_dir/output

# Remove files under specified directories.
rm test_dir/model/*
rm test_dir/output/*

# Print pwd and image name.
echo "current working directory is: $(pwd)"
echo "image name is: ${image}"

# Run docker container
# -v $(pwd)/test_dir:/opt/ml ---> Bind mount a volume. More info: https://docs.docker.com/engine/reference/commandline/run/#mount-volume--v---read-only
# --rm ---> Automatically remove the container when it exits.
# ${image} ---> Run image from ECR with specified image name.
# train ---> Last argument is which file will run at execution.
docker run -v $(pwd)/test_dir:/opt/ml --rm ${image} train
