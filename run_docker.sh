#!/bin/bash

WORK_DIR="/home/jay/git/SeeGround"
DATA_DIR="/media/TrainDataset"
LTDATA_DIR="/media/LTDataset/jay/seeground"

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
  --volume "${DATA_DIR}":/ws/data \
  --volume "${LTDATA_DIR}":/ws/LTdata \
  --volume /tmp/.X11-unix:/tmp/.X11-unix:rw \
  --volume /etc/localtime:/etc/localtime:ro \
  --volume /dev/input:/dev/input \
  --volume /dev/bus/usb:/dev/bus/usb:rw \
  --volume "${WORK_DIR}":/ws/external:rw \
  --workdir /ws/external \
  jayhi/seeground:3d-torch240-cu121 \
  bash
