#!/usr/bin/env bash
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

## @description  download nvidia docker bin
## @audience     public
## @stability    stable
## More details you can refer to https://github.com/NVIDIA/nvidia-docker/issues/655
## and https://github.com/NVIDIA/nvidia-docker/issues/635     
function download_nvidia_docker_bin()
{
  local NVIDIA_DOCKER_COMPONENTS=("libnvidia-container" "nvidia-container-runtime" "nvidia-docker")
  for component in "${NVIDIA_DOCKER_COMPONENTS[@]}"
  do
    if [[ ! -d "${DOWNLOAD_DIR}/nvidia-docker-repo/${component}" ]]; then
      mkdir -p "${DOWNLOAD_DIR}/nvidia-docker-repo/${component}"
    fi
    download_and_uncompress_nvidia_repo "${component}"
  done 
}

## @description  download and uncompress nvidia docker
## @audience     public
## @stability    stable
function download_and_uncompress_nvidia_repo()
{
  if [[ $# -ne 1 ]]; then
    echo -e "\\033[32mshell:> Failed to download nvidia-docker. 
      Please specify nvidia component for download_and_uncompress_nvidia_repo \\033[0m"
    return 1
  fi
  local component=$1

  if [[ -d "${DOWNLOAD_DIR}/nvidia-docker-repo/${component}/centos7" ]]; then
    echo "${DOWNLOAD_DIR}/nvidia-docker-repo/${component}/centos7 already exists."
  else
    # Trim the last slash of NVIDIA_DOCKER_GIT_SNAPSHOT_URL
    local NVIDIA_DOCKER_URL="$(echo -e "${NVIDIA_DOCKER_GIT_SNAPSHOT_URL}" | sed -e 's/\/*$//')"
    wget ${NVIDIA_DOCKER_URL}/${component}/tarball/gh-pages -O - | \
    tar -zx --strip-components=1 -C ${DOWNLOAD_DIR}/nvidia-docker-repo/${component}
    if [[ $? -ne 0 ]]; then
      echo -e "\\033[32mshell:> Failed to download ${component} of nvidia-docker 
        from ${NVIDIA_DOCKER_URL}/${component}/tarball/gh-pages \\033[0m"
    fi
  fi
}

## @description  install nvidia docker
## @audience     public
## @stability    stable
function install_nvidia_docker()
{
  # Backup /etc/docker/daemon.json
  local DOCKER_DAEMON_BAK="${DOWNLOAD_DIR}/docker-daemon-bak"
  if [[ ! -d "${DOCKER_DAEMON_BAK}" ]]; then
    mkdir -p "${DOCKER_DAEMON_BAK}"
  fi
  cp /etc/docker/daemon.json "${DOCKER_DAEMON_BAK}"
  echo "Backup /etc/docker/daemon.json in ${DOCKER_DAEMON_BAK}"

  # Remove nvidia docker 1.0
  remove_nvidia_docker_1.0
  
  # Get nvidia-docker repo
  if [[ ! -d "${DOWNLOAD_DIR}/nvidia-docker-repo" ]]; then
    mkdir -p "${DOWNLOAD_DIR}/nvidia-docker-repo"
  fi
  local dockerRepo="${DOWNLOAD_DIR}/nvidia-docker-repo/nvidia-docker.repo"
  if [[ -n "$DOWNLOAD_HTTP" ]]; then
    wget -P "${DOWNLOAD_DIR}/nvidia-docker-repo/" \
      "${DOWNLOAD_HTTP}/downloads/nvidia-docker-repo/nvidia-docker/centos7/nvidia-docker.repo"
    local DOWNLOAD_HTTP_REGEX=$(echo ${DOWNLOAD_HTTP} | sed 's/\//\\\//g')
    echo "DOWNLOAD_HTTP_REGEX: ${DOWNLOAD_HTTP_REGEX}"
    sed -i "s/https:\/\/nvidia.github.io/${DOWNLOAD_HTTP_REGEX}\/downloads\/nvidia-docker-repo/g" \
      "${dockerRepo}"
  else
    download_nvidia_docker_bin
    local DOWNLOAD_DIR_REGEX=$(echo "${DOWNLOAD_DIR}" | sed 's/\//\\\//g')
    cp "${DOWNLOAD_DIR}/nvidia-docker-repo/nvidia-docker/centos7/nvidia-docker.repo" \
      "${dockerRepo}"
    sed -i "s/https:\/\/nvidia.github.io/file:\/\/${DOWNLOAD_DIR_REGEX}\/nvidia-docker-repo/g" \
      "${dockerRepo}"
  fi

  # Install nvidia-docker
  sudo cp ${dockerRepo} /etc/yum.repos.d/nvidia-docker.repo
  echo -e "\\033[31m Installing nvidia-docker2 ...\\033[0m"
  sudo yum install -y nvidia-docker2-${NVIDIA_DOCKER_VERSION}-1.docker${DOCKER_VERSION_NUM}

  # As nvidia-docker would overwrite daemon.json, append old daemon.json into the now daemon.json
  COMBINE_JSON="${SCRIPTS_DIR}/combine-docker-daemons.py"
  IS_NEW_JSON=$(python ${COMBINE_JSON} ${DOCKER_DAEMON_BAK}/daemon.json /etc/docker/daemon.json ${DOCKER_DAEMON_BAK}/daemon-new.json)
  if [[ "${IS_NEW_JSON}" = "True" ]]; then
    sudo cp ${DOCKER_DAEMON_BAK}/daemon-new.json /etc/docker/daemon.json
    echo "Succeed to update /etc/docker/daemon.json"
  else 
    echo "WARNING: /etc/docker/daemon.json is overrided by nvidia-docker and
         can't be merged with the old daemon.json. Please update it manually
         later." 
  fi  

  # create nvidia driver library path
  if [ ! -d "/var/lib/nvidia-docker/volumes/nvidia_driver" ]; then
    echo "WARN: /var/lib/nvidia-docker/volumes/nvidia_driver folder path is not exist!"
    sudo mkdir -p /var/lib/nvidia-docker/volumes/nvidia_driver
  fi

  local nvidiaVersion
  nvidiaVersion=$(get_nvidia_version)
  echo -e "\\033[31m nvidia detect version is ${nvidiaVersion}\\033[0m"

  sudo mkdir "/var/lib/nvidia-docker/volumes/nvidia_driver/${nvidiaVersion}"
  sudo mkdir "/var/lib/nvidia-docker/volumes/nvidia_driver/${nvidiaVersion}/bin"
  sudo mkdir "/var/lib/nvidia-docker/volumes/nvidia_driver/${nvidiaVersion}/lib64"

  sudo cp /usr/bin/nvidia* "/var/lib/nvidia-docker/volumes/nvidia_driver/${nvidiaVersion}/bin"
  sudo cp /usr/lib64/libcuda* "/var/lib/nvidia-docker/volumes/nvidia_driver/${nvidiaVersion}/lib64"
  sudo cp /usr/lib64/libnvidia* "/var/lib/nvidia-docker/volumes/nvidia_driver/${nvidiaVersion}/lib64"

  echo -e "\\033[32m===== Please manually execute the following command =====\\033[0m"
  echo -e "\\033[32mshell:> nvidia-docker run --rm ${DOCKER_REGISTRY}/nvidia/cuda:9.0-devel nvidia-smi
# If you don't see the list of graphics cards above, the NVIDIA driver installation failed. =====
\\033[0m"

  echo -e "\\033[32m===== Please manually execute the following command =====\\033[0m"
  echo -e "\\033[32m# Test with tf.test.is_gpu_available()
shell:> nvidia-docker run -it ${DOCKER_REGISTRY}/tensorflow/tensorflow:1.9.0-gpu bash
# In docker container
container:> python
python:> import tensorflow as tf
python:> tf.test.is_gpu_available()
python:> exit()
\\033[0m"
}

## @description  uninstall nvidia docker
## @audience     public
## @stability    stable
function uninstall_nvidia_docker()
{
  sudo yum remove -y nvidia-docker2-${NVIDIA_DOCKER_VERSION}-1.docker${DOCKER_VERSION_NUM}
}

## @description  uninstall nvidia docker 1.0
## @audience     public
## @stability    stable
function remove_nvidia_docker_1.0()
{
  docker volume ls -q -f driver=nvidia-docker | \
    xargs -r -I{} -n1 docker ps -q -a -f volume={} | xargs -r docker rm -f
  sudo yum remove nvidia-docker
}
