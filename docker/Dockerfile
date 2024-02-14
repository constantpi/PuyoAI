#FROM ubuntu:22.04
FROM nvidia/cuda:12.0.0-base-ubuntu22.04
# FROM tensorflow/tensorflow:nightly-gpu

# Libraries you probably need
RUN apt-get update && apt-get install -y --no-install-recommends wget build-essential libreadline-dev \ 
libncursesw5-dev libssl-dev libsqlite3-dev libgdbm-dev libbz2-dev liblzma-dev zlib1g-dev uuid-dev libffi-dev libdb-dev

# python3.11 install
RUN wget --no-check-certificate https://www.python.org/ftp/python/3.11.4/Python-3.11.4.tgz \
&& tar -xf Python-3.11.4.tgz \
&& cd Python-3.11.4 \
&& ./configure --enable-optimizations\
&& make \
&& make install

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
    	vim \
        inetutils-ping \
        iproute2 \
        mesa-utils \
        nano \
        net-tools \
        openssh-client \
        zip \
        unzip \
        ca-certificates
# Remove unnecessary items to reduce size
RUN apt-get autoremove -y


# Install required python packages with pip
RUN pip3 install --upgrade pip
RUN pip3 install tqdm matplotlib
RUN pip3 install torch tensorboardX

RUN echo "source /root/main/docker/scripts/initialize-bash-shell.sh" >> /root/.bashrc

# Overwrite the entry point of the parent image.
ENTRYPOINT []

# Enter the container with a Bash shell.
CMD ["/bin/bash"]
