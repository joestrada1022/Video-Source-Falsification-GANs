# e an official TensorFlow GPU image as a base
FROM tensorflow/tensorflow:2.15.0-gpu-jupyter


# Install additional dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/*

# Install OpenCV using pip
RUN pip install opencv-contrib-python tqdm pandas seaborn

