#!/bin/bash

WORK_DIR="/home/jay/git/SeeGround"

# Docker run
docker run -it --rm \
  --gpus all \
  --shm-size=32g \
  --net=host \
  --privileged \
  --name "seeground" \
  --env DISPLAY \
  --env QT_X11_NO_MITSHM=1 \
  --env XAUTHORITY=/tmp/.docker.xauth \
  --volume /media/jay/T9/TrainDataset/:/ws/data \
  --volume /tmp/.X11-unix:/tmp/.X11-unix:rw \
  --volume /etc/localtime:/etc/localtime:ro \
  --volume /dev/input:/dev/input \
  --volume /dev/bus/usb:/dev/bus/usb:rw \
  --volume "${WORK_DIR}":/ws/external:rw \
  --workdir /ws/external \
  qwenllm/qwenvl \
  bash