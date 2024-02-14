#!/bin/bash

################################################################################

# Link the default shell 'sh' to Bash.
alias sh='/bin/bash'
alias python=python3
alias pip=pip3

################################################################################

# Configure the terminal.

# Disable flow control. If enabled, inputting 'ctrl+s' locks the terminal until inputting 'ctrl+q'.
stty -ixon

################################################################################

# Configure 'umask' for giving read/write/execute permission to group members.
umask 0002

################################################################################
export PS1="\[[44;1;37m\]<root>\[[0m\]\w$"
export DOCKER_SCRIPTS_PATH=/root/main/docker/scripts

################################################################################
shopt -s dotglob  #ワイルドカード*で隠しファイルも含まれるようにする。これをやめたいなら-uにする

################################################################################

# Move to the working directory.
cd /root/main/src
alias rm=trash-put
