#!/bin/bash

echo "Running Training Script Inside GPU-Accelerated Container"
# run the command in the container
docker compose up -d
# interactive shell
docker compose exec jupyter python src/train_wgan.py \
 --data_path data/frames/ --image_path generated/images/wcgan-gp/ \
 --model_path generated/models/wcgan-gp --tensorboard_path generated/tensors/wcgan-gp \
#  --use_cpu True   # uncomment this line to run on CPU

docker compose down