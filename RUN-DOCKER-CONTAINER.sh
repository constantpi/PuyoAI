#!/bin/bash

if [ -e /.dockerenv ]; then
  echo "This script is running on Docker container."
  exit 1
fi

cd $(dirname $0)
################################################################################
SERVICE_NAME=puyo

PROJECT=$(whoami)

CONTAINER="${PROJECT}_${SERVICE_NAME}_1"
export COMPOSE_FILE="./docker/docker-compose.yml"
if [ ! -z "$(which nvidia-smi)" ]; then
  export COMPOSE_FILE=${COMPOSE_FILE}:./docker/docker-compose-gpu.yml
fi
echo "$0: COMPOSE_FILE=${COMPOSE_FILE}"
echo "$0: PROJECT=${PROJECT}"
echo "$0: CONTAINER=${CONTAINER}"


# Run the Docker container in the background.
# Any changes made to './docker-compose.yml' will recreate and overwrite the container.

EXIST_CONTAINER=$(docker ps |grep $CONTAINER)
if [ -z "$EXIST_CONTAINER" ]; then
  docker-compose -p ${PROJECT} up -d
  docker exec -it ${CONTAINER} bash -c "cp ~/main/docker/scripts/gitconfig ~/.gitconfig"
else
  echo "${CONTAINER}はすでに存在します"
fi

################################################################################
docker exec -i -t ${CONTAINER} bash

