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

## @description  check install user
## @audience     public
## @stability    stable
function check_install_user()
{
  if [[ $(id -u) -ne 0 ]];then
    echo "This script must be run with a ROOT user!"
    exit # don't call exit_install()
  fi
}

## @description  exit install
## @audience     public
## @stability    stable
function exit_install()
{
  echo "Exit the installation!"
  exit
}

## @description  Check if the IP address format is correct
## @audience     public
## @stability    stable
function valid_ip()
{
  local ip=$1
  local stat=1

  if [[ $ip =~ ^[0-9]{1,3\}.[0-9]{1,3\}.[0-9]{1,3\}.[0-9]{1,3\}$ ]]; then
    OIFS=$IFS
    IFS='.'
    ip=($ip)
    IFS=$OIFS

    if [[ ${ip[0]} -le 255 && ${ip[1]} -le 255 && ${ip[2]} -le 255 && ${ip[3]} -le 255 ]]; then
      stat=$?
    fi
  fi

  return $stat
}

## @description  Check if the configuration file configuration is correct
## @audience     public
## @stability    stable
function check_install_conf()
{
  echo "Check if the configuration file configuration is correct ..."

  # check YARN_SECURITY's value 
    if [ -z "${YARN_SECURITY}" ]; then
    echo "YARN_SECURITY=[$YARN_SECURITY] is empty! please use true or false "
    exit_install
    fi

  # check resource manager
  rmCount=${#YARN_RESOURCE_MANAGER_HOSTS[@]}
  if [[ $rmCount -gt 2 ]]; then # <>2
    echo "Number of resource manager nodes = [$rmCount], must be configured equal 2 servers! "
    exit_install
  fi

  # 
  if [ -z "${YARN_REGISTRY_DNS_HOST_PORT}" ]; then
    echo "YARN_REGISTRY_DNS_HOST_PORT=[$YARN_REGISTRY_DNS_HOST_PORT] is empty! "
    exit_install
  fi

  # Check if it is empty
  if [[ "${YARN_SECURITY}" = "true" && -z "${LOCAL_REALM}" ]]; then
    echo "LOCAL_REALM=[$LOCAL_REALM] can not be empty! "
    exit_install
  fi

  if [[ "${YARN_SECURITY}" = "true" && -z "${HADOOP_KEYTAB_LOCATION}" ]]; then
    echo "HADOOP_KEYTAB_LOCATION=[$HADOOP_KEYTAB_LOCATION] can not be empty! "
    exit_install
  fi

  if [[ "${YARN_SECURITY}" = "true" && -z "${HADOOP_PRINCIPAL}" ]]; then
    echo "HADOOP_PRINCIPAL=[$HADOOP_PRINCIPAL] can not be empty! "
    exit_install
  fi

  if [[ "${YARN_SECURITY}" = "true" && -z "${MAPRED_KEYTAB_LOCATION}" ]]; then
    echo "MAPRED_KEYTAB_LOCATION=[$MAPRED_KEYTAB_LOCATION] can not be empty! "
    exit_install
  fi

  if [[ "${YARN_SECURITY}" = "true" && -z "${YARN_KEYTAB_LOCATION}" ]]; then
    echo "YARN_KEYTAB_LOCATION=[$YARN_KEYTAB_LOCATION] can not be empty! "
    exit_install
  fi

  if [[ "${YARN_SECURITY}" = "true" && -z "${HTTP_KEYTAB_LOCATION}" ]]; then
    echo "HTTP_KEYTAB_LOCATION=[$HTTP_KEYTAB_LOCATION] can not be empty! "
    exit_install
  fi

  # check etcd conf
  hostCount=${#ETCD_HOSTS[@]}
  if [[ $hostCount -lt 3 && hostCount -ne 0 ]]; then # <>2
    echo "Number of nodes = [$hostCount], must be configured to be greater than or equal to 3 servers! "
    exit_install
  fi
  echo "Check if the configuration file configuration is correct [ Done ]"
}

## @description  index by EtcdHosts list
## @audience     public
## @stability    stable
function indexByEtcdHosts() {
  index=0
  while [ "$index" -lt "${#ETCD_HOSTS[@]}" ]; do
    if [ "${ETCD_HOSTS[$index]}" = "$1" ]; then
      echo $index
      return
    fi
    (( index++ ))
  done
  echo ""
}

## @description  index by Resource Manager Hosts list
## @audience     public
## @stability    stable
function indexByRMHosts() {
  index=0
  while [ "$index" -lt "${#YARN_RESOURCE_MANAGER_HOSTS[@]}" ]; do
    if [ "${YARN_RESOURCE_MANAGER_HOSTS[$index]}" = "$1" ]; then
      echo $index
      return
    fi
    (( index++ ))
  done
  echo ""
}

## @description  index of node manager exclude Hosts list
## @audience     public
## @stability    stable
function indexOfNMExcludeHosts() {
  index=0
  while [ "$index" -lt "${#YARN_NODE_MANAGER_EXCLUDE_HOSTS[@]}" ]; do
    if [ "${YARN_NODE_MANAGER_EXCLUDE_HOSTS[$index]}" = "$1" ]; then
      echo $index
      return
    fi
    (( index++ ))
  done
  echo ""
}

## @description  index by Resource Manager Hosts list
## @audience     public
## @stability    stable
function pathExitsOnHDFS() {
  exists=$("${HADOOP_HOME}/bin/hadoop" dfs -ls -d "$1")
  echo "${exists}"
}

## @description  get local IP
## @audience     public
## @stability    stable
function getLocalIP()
{
  local _ip _myip _line _nl=$'\n'
  while IFS=$': \t' read -r -a _line ;do
      [ -z "${_line%inet}" ] &&
         _ip=${_line[${#_line[1]}>4?1:2]} &&
         [ "${_ip#127.0.0.1}" ] && _myip=$_ip
    done< <(LANG=C /sbin/ifconfig)
  printf "%s" ${1+-v} "$1" "%s${_nl:0:$((${#1}>0?0:1))}" "$_myip"
}

## @description  get ip list
## @audience     public
## @stability    stable
function get_ip_list()
{
  array=$(ip -o -4 addr | awk '{print $4}' | grep -v 127 | cut -d/ -f1)

  for ip in "${array[@]}"
  do
    LOCAL_HOST_IP_LIST+=(${ip})
  done
}
