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

## @description  download docker rmp
## @audience     public
## @stability    stable
function download_docker_rpm()
{
  # download http server
  if [[ -n "$DOWNLOAD_HTTP" ]]; then
    MY_DOCKER_ENGINE_RPM="${DOWNLOAD_HTTP}/downloads/docker/${DOCKER_ENGINE_RPM}"
  else
    # Trim the last slash of DOCKER_REPO
    DOCKER_REPO_TRIMED="$(echo -e "${DOCKER_REPO}" | sed -e 's/\/*$//')"
    MY_DOCKER_ENGINE_RPM=${DOCKER_REPO_TRIMED}/${DOCKER_ENGINE_RPM}
  fi

  if [[ -f ${DOWNLOAD_DIR}/docker/${DOCKER_ENGINE_RPM} ]]; then
    echo "${DOWNLOAD_DIR}/docker/${DOCKER_ENGINE_RPM} already exists."
  else
    echo "download ${MY_DOCKER_ENGINE_RPM} ..."
    wget -P "${DOWNLOAD_DIR}/docker/" "${MY_DOCKER_ENGINE_RPM}"
    if [[ $? -ne 0 ]]; then
      echo -e "\\033[32mshell:> Failed to download ${DOCKER_ENGINE_RPM} of docker 
        from ${MY_DOCKER_ENGINE_RPM} \\033[0m"
    fi
  fi
}

## @description  install docker bin
## @audience     public
## @stability    stable
function install_docker_bin()
{
  download_docker_rpm

  sudo yum -y localinstall "${DOWNLOAD_DIR}/docker/${DOCKER_ENGINE_RPM}"
}

## @description  uninstall docker bin
## @audience     public
## @stability    stable
function uninstall_docker_bin()
{
  sudo yum -y remove "${DOCKER_VERSION}"
}

## @description  install docker config
## @audience     public
## @stability    stable
function install_docker_config()
{
  rm -rf "${INSTALL_TEMP_DIR}/docker"
  cp -rf "${PACKAGE_DIR}/docker" "${INSTALL_TEMP_DIR}/"

  # replace cluster-store
  # "cluster-store":"etcd://10.196.69.173:2379,10.196.69.174:2379,10.196.69.175:2379"
  # char '/' need to escape '\/'
  clusterStore="etcd:\\/\\/"
  index=1
  etcdHostsSize=${#ETCD_HOSTS[@]}
  for item in "${ETCD_HOSTS[@]}"
  do
    clusterStore="${clusterStore}${item}:2379"
    if [[ ${index} -lt ${etcdHostsSize} ]]; then
      clusterStore=${clusterStore}","
    fi
    index=$((index+1))
  done
  # echo "clusterStore=${clusterStore}"
  sed -i "s/CLUSTER_STORE_REPLACE/${clusterStore}/g" "$INSTALL_TEMP_DIR/docker/daemon.json"

  sed -i "s/DOCKER_REGISTRY_REPLACE/${DOCKER_REGISTRY}/g" "$INSTALL_TEMP_DIR/docker/daemon.json"
  sed -i "s/LOCAL_HOST_IP_REPLACE/${LOCAL_HOST_IP}/g" "$INSTALL_TEMP_DIR/docker/daemon.json"
  YARN_REGISTRY_DNS_IP=$(getent hosts "${YARN_REGISTRY_DNS_HOST}" | awk '{ print $1 }')
  sed -i "s/YARN_REGISTRY_DNS_HOST_REPLACE/${YARN_REGISTRY_DNS_IP}/g" "$INSTALL_TEMP_DIR/docker/daemon.json"
  
  hosts=${LOCAL_DNS_HOST//,/ }
  hosts_length=0
  for element in $hosts
  do
    hosts_length=$((${hosts_length} + 1))
    if [ ${hosts_length} != 1 ]; then
      NEW_LOCAL_DNS_HOST="${NEW_LOCAL_DNS_HOST}, "
    fi
      NEW_LOCAL_DNS_HOST="${NEW_LOCAL_DNS_HOST}\"${element}\""
  done
  sed -i "s/LOCAL_DNS_HOST_REPLACE/${NEW_LOCAL_DNS_HOST}/g" "$INSTALL_TEMP_DIR/docker/daemon.json"

  # Delete the ASF license comment in the daemon.json file, otherwise it will cause a json format error.
  sed -i '1,16d' "$INSTALL_TEMP_DIR/docker/daemon.json"

  if [ ! -d "/etc/docker" ]; then
    sudo mkdir /etc/docker
  fi

  sudo cp "$INSTALL_TEMP_DIR/docker/daemon.json" /etc/docker/
  sudo cp "$INSTALL_TEMP_DIR/docker/docker.service" /etc/systemd/system/

  # Change docker store path
  if [[ -n "${DOCKER_STORE_PATH}" ]]; then
    mkdir -p "${DOCKER_STORE_PATH}"
    cp -r /var/lib/docker/* "${DOCKER_STORE_PATH}"
    rm -rf /var/lib/docker
    ln -s "${DOCKER_STORE_PATH}" "/var/lib/docker"
  fi
}

## @description  install docker
## @audience     public
## @stability    stable
function install_docker()
{
  install_docker_bin
  install_docker_config

  sudo systemctl daemon-reload
  sudo systemctl enable docker.service
}

## @description  unstall docker
## @audience     public
## @stability    stable
function uninstall_docker()
{
  echo "stop docker service"
  sudo systemctl stop docker

  echo "remove docker"
  uninstall_docker_bin

  sudo rm /etc/systemd/system/docker.service

  sudo systemctl daemon-reload
}

## @description  start docker
## @audience     public
## @stability    stable
function start_docker()
{
  sudo systemctl restart docker
  sudo systemctl status docker
  docker info
}

## @description  stop docker
## @audience     public
## @stability    stable
function stop_docker()
{
  sudo systemctl stop docker
  sudo systemctl status docker
}

## @description  check if the containers exist
## @audience     public
## @stability    stable
function containers_exist()
{
  local dockerContainersInfo
  dockerContainersInfo=$(docker ps -a --filter NAME="$1")
  echo "${dockerContainersInfo}" | grep "$1"
}
