#!/bin/bash

# check if arguments are provided
if [ -z "$1" ]; then
    echo "Usage: $0 <cmd> <args>"
    exit 1
fi

# run any cmd in the gpu-accelerated container
echo "Running $1 $2 in the container"

# run the command in the container
docker compose up -d
# interactive shell
docker compose exec jupyter python $1 $2
docker compose down