#!/bin/bash

if [ -e /.dockerenv ]; then
  echo "This script is running on Docker container."
  exit 1
fi

################################################################################

SERVICE_NAME=puyo

PROJECT=$(whoami)

CONTAINER="${PROJECT}_${SERVICE_NAME}_1"
echo "$0: PROJECT=${PROJECT}"
echo "$0: CONTAINER=${CONTAINER}"

# Stop and remove the Docker container.
EXISTING_CONTAINER_ID=`docker ps -aq -f name=${CONTAINER}`
if [ ! -z "${EXISTING_CONTAINER_ID}" ]; then
  # echo "Stop the container ${CONTAINER} with ID: ${EXISTING_CONTAINER_ID}."
  # docker stop ${EXISTING_CONTAINER_ID}
  # echo "Remove the container ${CONTAINER} with ID: ${EXISTING_CONTAINER_ID}."
  # docker rm ${EXISTING_CONTAINER_ID}
  echo "The container name ${CONTAINER} is already in use" 1>&2
  echo ${EXISTING_CONTAINER_ID}
  exit 1
fi

################################################################################

# Build the Docker image with the Nvidia GL library.
echo "starting build"

export COMPOSE_FILE="./docker/docker-compose.yml"
echo "$0: COMPOSE_FILE=${COMPOSE_FILE}"
docker-compose -p ${PROJECT} build
