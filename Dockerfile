# e an official TensorFlow GPU image as a base
FROM tensorflow/tensorflow:latest-gpu-jupyter


# Install additional dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/*

# Install OpenCV using pip
RUN pip install opencv-contrib-python tqdm

