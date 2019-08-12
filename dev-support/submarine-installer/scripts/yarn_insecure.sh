#!/bin/bash
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

## @description  install yarn
## @audience     public
## @stability    stable
function install_yarn_insecure()
{
  # change the soft link of etc   
  rm -f ${PACKAGE_DIR}/hadoop/yarn/etc
  ln -s ${PACKAGE_DIR}/hadoop/yarn/etc_insecure ${PACKAGE_DIR}/hadoop/yarn/etc

  initialize_temp_insecure

  host=$(hostname)
  index=$(indexByRMHosts "${host}")
  #if [[ -n "$index" || "x$YARN_TIMELINE_HOST" != "x$host" ]]; then
  #  kinit -kt ${HADOOP_KEYTAB_LOCATION} ${HADOOP_PRINCIPAL}
  #fi

  install_java_tarball
  if [[ $? = 1 ]]; then
    return 1
  fi
  install_yarn_tarball_insecure
  if [[ $? = 1 ]]; then
    return 1
  fi
  install_yarn_sbin_insecure
  install_yarn_rm_nm_insecure
  install_yarn_service_insecure
  install_registery_dns_insecure
  install_timeline_server_insecure
  install_job_history_insecure
  install_mapred_insecure
  install_spark_suffle_insecure
  install_lzo_native_insecure

  # copy file
  cp -f "$INSTALL_TEMP_DIR/hadoop/yarn/etc/hadoop/yarn-site.xml" "${HADOOP_HOME}/etc/hadoop/"
  cp -f "$INSTALL_TEMP_DIR/hadoop/yarn/etc/hadoop/mapred-site.xml" "${HADOOP_HOME}/etc/hadoop/"
  cp -f "$INSTALL_TEMP_DIR/hadoop/yarn/etc/hadoop/core-site.xml" "${HADOOP_HOME}/etc/hadoop/"
  cp -f "$INSTALL_TEMP_DIR/hadoop/yarn/etc/hadoop/hdfs-site.xml" "${HADOOP_HOME}/etc/hadoop/"
  cp -f "$INSTALL_TEMP_DIR/hadoop/yarn/etc/hadoop/capacity-scheduler.xml" "${HADOOP_HOME}/etc/hadoop/"
  cp -f "$INSTALL_TEMP_DIR/hadoop/yarn/etc/hadoop/resource-types.xml" "${HADOOP_HOME}/etc/hadoop/"
  cp -f "$INSTALL_TEMP_DIR/hadoop/yarn/etc/hadoop/log4j.properties" "${HADOOP_HOME}/etc/hadoop/"
  chown "${HADOOP_SETUP_USER}":yarn "${HADOOP_HOME}"/etc/hadoop/*
}

## @description  Initialize tmp dir for installation.
## @audience     public
## @stability    stable
function initialize_temp_insecure()
{
  mkdir -p "${INSTALL_TEMP_DIR}/hadoop"
  \cp -rf "${PACKAGE_DIR}/hadoop/yarn" "${INSTALL_TEMP_DIR}/hadoop/"
  isGpuEnabled=$(nvidia-smi)
  if [[ -n "$isGpuEnabled" ]]; then
    python ${SCRIPTS_DIR}/xmlcombine.py ${PACKAGE_DIR}/hadoop/yarn/etc/hadoop/yarn-site.xml ${PACKAGE_DIR}/hadoop/yarn/etc/hadoop/gpu/yarn-site-gpu.xml > "${INSTALL_TEMP_DIR}/hadoop/yarn/etc/hadoop/yarn-site.xml"
  fi
  chown -R ${HADOOP_SETUP_USER} "${INSTALL_TEMP_DIR}/hadoop"
}

## @description  uninstall yarn
## @audience     public
## @stability    stable
function uninstall_yarn()
{
  executor_dir=$(dirname "${YARN_CONTAINER_EXECUTOR_PATH}")
  executor_conf_dir=$(dirname "${executor_dir}")/etc/hadoop

  rm -rf "${YARN_CONTAINER_EXECUTOR_PATH}"
  rm -rf "${executor_conf_dir}"
  rm -rf "${HADOOP_HOME}"
}

## @description  install yarn container executor
## @audience     public
## @stability    stable
function install_yarn_container_executor_insecure()
{
  echo "install yarn container executor file ..."

  executor_dir=$(dirname "${YARN_CONTAINER_EXECUTOR_PATH}")
  if [ ! -d "${executor_dir}" ]; then
    mkdir -p "${executor_dir}"
  fi
  if [ -f "${YARN_CONTAINER_EXECUTOR_PATH}" ]; then
    if [ -f "${HADOOP_HOME}/bin/container-executor" ]; then
      rm ${YARN_CONTAINER_EXECUTOR_PATH}
    fi
  fi

  if [ -f "${HADOOP_HOME}/bin/container-executor" ]; then
    cp -f "${HADOOP_HOME}/bin/container-executor" "${YARN_CONTAINER_EXECUTOR_PATH}"
    rm "${HADOOP_HOME}/bin/container-executor"
  fi
  
  sudo chmod 6755 "${executor_dir}"
  sudo chown :yarn "${YARN_CONTAINER_EXECUTOR_PATH}"
  sudo chmod 6050 "${YARN_CONTAINER_EXECUTOR_PATH}"
}

## @description  Deploy hadoop yarn tar ball
## @audience     public
## @stability    stable
function install_yarn_tarball_insecure()
{
  tag=`date '+%Y%m%d%H%M%S'`
  if [ -f "${PACKAGE_DIR}/hadoop/${HADOOP_TARBALL}" ]; then
    tar -zxvf "${PACKAGE_DIR}/hadoop/${HADOOP_TARBALL}" -C "${PACKAGE_DIR}/hadoop/"
    mv "${PACKAGE_DIR}/hadoop/${HADOOP_VERSION}" "/home/${HADOOP_SETUP_USER}/${HADOOP_VERSION}-${tag}"
    chown -R ${HADOOP_SETUP_USER} "/home/hadoop/${HADOOP_VERSION}-${tag}"
    if [[ -d "${HADOOP_HOME}" ]] || [[ -L "${HADOOP_HOME}" ]]; then
      rm -rf ${HADOOP_HOME}
    fi
    ln -s "/home/hadoop/${HADOOP_VERSION}-${tag}" "${HADOOP_HOME}"
    chown ${HADOOP_SETUP_USER} "${HADOOP_HOME}"
  else
    echo "ERROR: Please put ${HADOOP_TARBALL} in the path of ${PACKAGE_DIR}/hadoop/ fristly."
    return 1
  fi
}

## @description  Deploy java tar ball
## @audience     public
## @stability    stable
function install_java_tarball()
{
  if [[ -d "${JAVA_HOME}" ]] || [[ -L "${JAVA_HOME}" ]]; then
    echo "JAVA_HOME already exists. There is no need to install java."
  else 
    if [[ -f "${PACKAGE_DIR}/java/${JAVA_TARBALL}" ]]; then
      tar -zxvf "${PACKAGE_DIR}/java/${JAVA_TARBALL}" -C "${PACKAGE_DIR}/java/"
      mv "${PACKAGE_DIR}/java/${JAVA_VERSION}" "/home/${HADOOP_SETUP_USER}/${JAVA_VERSION}"
      chown -R ${HADOOP_SETUP_USER} "/home/hadoop/${JAVA_VERSION}"
      ln -s "/home/hadoop/${JAVA_VERSION}" "${JAVA_HOME}" 
    else
      echo "Error: Failed to install java, please put java tallball in the path of
        ${PACKAGE_DIR}/java/${JAVA_TARBALL}"
      return 1
    fi
  fi
}

## @description  install yarn resource & node manager
## @audience     public
## @stability    stable
function install_yarn_rm_nm_insecure()
{
  echo "install yarn config file ..."
  host=$(hostname)

  find="/"
  replace="\\/"
  escape_yarn_nodemanager_local_dirs=${YARN_NODEMANAGER_LOCAL_DIRS//$find/$replace}
  escape_yarn_nodemanager_log_dirs=${YARN_NODEMANAGER_LOG_DIRS//$find/$replace}
  escape_yarn_hierarchy=${YARN_HIERARCHY//$find/$replace}
  escape_yarn_nodemanager_nodes_exclude_path=${YARN_RESOURCEMANAGER_NODES_EXCLUDE_PATH//$find/$replace}
  escape_yarn_nodemanager_recovery_dir=${YARN_NODEMANAGER_RECOVERY_DIR//$find/$replace}
  escape_fs_defaults=${FS_DEFAULTFS//$find/$replace}

  # container-executor.cfg`
  sed -i "s/YARN_NODEMANAGER_LOCAL_DIRS_REPLACE/${escape_yarn_nodemanager_local_dirs}/g" "$INSTALL_TEMP_DIR/hadoop/yarn/etc/hadoop/container-executor.cfg"
  sed -i "s/YARN_NODEMANAGER_LOG_DIRS_REPLACE/${escape_yarn_nodemanager_log_dirs}/g" "$INSTALL_TEMP_DIR/hadoop/yarn/etc/hadoop/container-executor.cfg"
  sed -i "s/DOCKER_REGISTRY_REPLACE/${DOCKER_REGISTRY}/g" "$INSTALL_TEMP_DIR/hadoop/yarn/etc/hadoop/container-executor.cfg"
  sed -i "s/CALICO_NETWORK_NAME_REPLACE/${CALICO_NETWORK_NAME}/g" "$INSTALL_TEMP_DIR/hadoop/yarn/etc/hadoop/container-executor.cfg"
  sed -i "s/YARN_HIERARCHY_REPLACE/${escape_yarn_hierarchy}/g" "$INSTALL_TEMP_DIR/hadoop/yarn/etc/hadoop/container-executor.cfg"

  # enable cgroup for yarn
  . "${PACKAGE_DIR}/submarine/submarine.sh"

  # Delete the ASF license comment in the container-executor.cfg file, otherwise it will cause a cfg format error.
  sed -i '1,16d' "$INSTALL_TEMP_DIR/hadoop/yarn/etc/hadoop/container-executor.cfg"
  
  executor_dir=$(dirname "${YARN_CONTAINER_EXECUTOR_PATH}")
  executor_conf_dir=$(dirname "${executor_dir}")/etc/hadoop
  if [ ! -d "${executor_conf_dir}" ]; then
    sudo mkdir -p "${executor_conf_dir}"
  fi

  cp -f "$INSTALL_TEMP_DIR/hadoop/yarn/etc/hadoop/container-executor.cfg" "${executor_conf_dir}" 

  # yarn-site.xml
  sed -i "s/YARN_RESOURCE_MANAGER_HOSTS1_REPLACE/${YARN_RESOURCE_MANAGER_HOSTS[0]}/g" "$INSTALL_TEMP_DIR/hadoop/yarn/etc/hadoop/yarn-site.xml"
  sed -i "s/YARN_RESOURCE_MANAGER_HOSTS2_REPLACE/${YARN_RESOURCE_MANAGER_HOSTS[1]}/g" "$INSTALL_TEMP_DIR/hadoop/yarn/etc/hadoop/yarn-site.xml"
  sed -i "s/LOCAL_CLUSTER_ID_REPLACE/${LOCAL_CLUSTER_ID}/g" "$INSTALL_TEMP_DIR/hadoop/yarn/etc/hadoop/yarn-site.xml"
  sed -i "s/YARN_NODEMANAGER_LOCAL_DIRS_REPLACE/${escape_yarn_nodemanager_local_dirs}/g" "$INSTALL_TEMP_DIR/hadoop/yarn/etc/hadoop/yarn-site.xml"
  sed -i "s/YARN_NODEMANAGER_LOG_DIRS_REPLACE/${escape_yarn_nodemanager_log_dirs}/g" "$INSTALL_TEMP_DIR/hadoop/yarn/etc/hadoop/yarn-site.xml"
  # Make nodemanager local dirs if the host is not in YARN_NODE_MANAGER_EXCLUDE_HOSTS
  index=$(indexOfNMExcludeHosts "${host}")
  if [ -z "$index" ]; then
    arr=(${YARN_NODEMANAGER_LOCAL_DIRS//,/ })
    index=0
    while [ "$index" -lt "${#arr[@]}" ]; do
      mkdir -p "${arr[$index]}"
      (( index++ ))
    done

    arr=(${YARN_NODEMANAGER_LOG_DIRS//,/ })
    index=0
    while [ "$index" -lt "${#arr[@]}" ]; do
      mkdir -p "${arr[$index]}"
      (( index++ ))
    done

    arr=(${YARN_NODEMANAGER_LOCAL_HOME_PATHS//,/ })
    index=0
    while [ "$index" -lt "${#arr[@]}" ]; do
      chown -R yarn "${arr[$index]}"
      (( index++ ))
    done
  fi
 
  sed -i "s/YARN_ZK_ADDRESS_REPLACE/${YARN_ZK_ADDRESS}/g" "$INSTALL_TEMP_DIR/hadoop/yarn/etc/hadoop/yarn-site.xml"
  sed -i "s/CALICO_NETWORK_NAME_REPLACE/${CALICO_NETWORK_NAME}/g" "$INSTALL_TEMP_DIR/hadoop/yarn/etc/hadoop/yarn-site.xml"

  sed -i "s/YARN_RESOURCEMANAGER_NODES_EXCLUDE_PATH_REPLACE/${escape_yarn_nodemanager_nodes_exclude_path}/g" "$INSTALL_TEMP_DIR/hadoop/yarn/etc/hadoop/yarn-site.xml"
  node_exclude_dir=$(dirname "${YARN_RESOURCEMANAGER_NODES_EXCLUDE_PATH}")
  if [[ ! -d "${node_exclude_dir}" ]]; then
    mkdir -p "${YARN_RESOURCEMANAGER_NODES_EXCLUDE_PATH}"
  fi
  if [[ ! -f "${YARN_RESOURCEMANAGER_NODES_EXCLUDE_PATH}" ]]; then
    touch "${YARN_RESOURCEMANAGER_NODES_EXCLUDE_PATH}"
    chmod 777 "${YARN_RESOURCEMANAGER_NODES_EXCLUDE_PATH}"
  fi

  sed -i "s/YARN_NODEMANAGER_RECOVERY_DIR_REPLACE/${escape_yarn_nodemanager_recovery_dir}/g" "$INSTALL_TEMP_DIR/hadoop/yarn/etc/hadoop/yarn-site.xml"
  mkdir -p "${YARN_NODEMANAGER_RECOVERY_DIR}"
  chmod 777 "${YARN_NODEMANAGER_RECOVERY_DIR}"
  
  # core-site.xml
  sed -i "s/YARN_ZK_ADDRESS_REPLACE/${YARN_ZK_ADDRESS}/g" "$INSTALL_TEMP_DIR/hadoop/yarn/etc/hadoop/core-site.xml"
  sed -i "s/FS_DEFAULTFS_REPLACE/${escape_fs_defaults}/g" "$INSTALL_TEMP_DIR/hadoop/yarn/etc/hadoop/core-site.xml"

  # WARN: ${HADOOP_HTTP_AUTHENTICATION_SIGNATURE_SECRET_FILE} Can not be empty!
  echo 'hello submarine' > "${HADOOP_HTTP_AUTHENTICATION_SIGNATURE_SECRET_FILE}"
  escape_hadoop_http_authentication_signature_secret_file=${HADOOP_HTTP_AUTHENTICATION_SIGNATURE_SECRET_FILE//$find/$replace}
  sed -i "s/HADOOP_HTTP_AUTHENTICATION_SIGNATURE_SECRET_FILE_REPLACE/${escape_hadoop_http_authentication_signature_secret_file}/g" "$INSTALL_TEMP_DIR/hadoop/yarn/etc/hadoop/core-site.xml"

  cp -f "$INSTALL_TEMP_DIR/hadoop/yarn/etc/hadoop/core-site.xml" "${HADOOP_HOME}/etc/hadoop/"
  cp -f "$INSTALL_TEMP_DIR/hadoop/yarn/etc/hadoop/hdfs-site.xml" "${HADOOP_HOME}/etc/hadoop/"

  install_yarn_container_executor_insecure
}

function install_spark_suffle_insecure() {
  cp -R ${PACKAGE_DIR}/hadoop/yarn/lib/spark* "${HADOOP_HOME}/share/hadoop/yarn/lib/"
}

function install_lzo_native_insecure() {
  cp -R ${PACKAGE_DIR}/hadoop/yarn/lib/native/libgpl* "${HADOOP_HOME}/lib/native/"
  cp -R ${PACKAGE_DIR}/hadoop/yarn/lib/hadoop-lzo* "${HADOOP_HOME}/share/hadoop/yarn/lib/" 
  cp -R ${PACKAGE_DIR}/hadoop/yarn/lib/hadoop-lzo* "${HADOOP_HOME}/share/hadoop/hdfs/lib/" 
  cp -R ${PACKAGE_DIR}/hadoop/yarn/lib/hadoop-lzo* "${HADOOP_HOME}/share/hadoop/common/lib/"
  if [ ! -d "${HADOOP_HOME}/share/hadoop/mapreduce/lib/" ]; then
    mkdir -p "${HADOOP_HOME}/share/hadoop/mapreduce/lib/"
  fi 
  cp -R ${PACKAGE_DIR}/hadoop/yarn/lib/hadoop-lzo* "${HADOOP_HOME}/share/hadoop/mapreduce/lib/" 
}

function install_mapred_insecure() {
  find="/"
  replace="\\/"
  escape_yarn_app_mapreduce_am_staging_dir=${YARN_APP_MAPREDUCE_AM_STAGING_DIR//$find/$replace}
  escape_fs_defaults=${FS_DEFAULTFS//$find/$replace}
 
  sed -i "s/FS_DEFAULTFS_REPLACE/${escape_fs_defaults}/g" "$INSTALL_TEMP_DIR/hadoop/yarn/etc/hadoop/mapred-site.xml" 
  sed -i "s/YARN_APP_MAPREDUCE_AM_STAGING_DIR_REPLACE/${escape_yarn_app_mapreduce_am_staging_dir}/g" "$INSTALL_TEMP_DIR/hadoop/yarn/etc/hadoop/mapred-site.xml"
  host=$(hostname)
  index=$(indexByRMHosts "${host}")
  if [[ -n "$index" ]]; then
    # Only RM needs to execute the following code
    pathExists=$(pathExitsOnHDFS "${YARN_APP_MAPREDUCE_AM_STAGING_DIR}")
    if [[ -z "${pathExists}" ]]; then
      echo "Create hdfs path ${YARN_APP_MAPREDUCE_AM_STAGING_DIR}"
      "${HADOOP_HOME}/bin/hadoop" dfs -mkdir -p "${YARN_APP_MAPREDUCE_AM_STAGING_DIR}"
      "${HADOOP_HOME}/bin/hadoop" dfs -chown yarn:hadoop "${YARN_APP_MAPREDUCE_AM_STAGING_DIR}"
      "${HADOOP_HOME}/bin/hadoop" dfs -chmod 1777 "${YARN_APP_MAPREDUCE_AM_STAGING_DIR}"
    fi
  fi
}

function install_yarn_sbin_insecure() {
  find="/"
  replace="\\/"
  escape_yarn_gc_log_dir=${YARN_GC_LOG_DIR//$find/$replace}
  escape_java_home=${JAVA_HOME//$find/$replace}
  escape_hadoop_home=${HADOOP_HOME//$find/$replace}
  escape_yarn_pid_dir=${YARN_PID_DIR//$find/$replace}
  escape_yarn_log_dir=${YARN_LOG_DIR//$find/$replace}
  cp -R ${PACKAGE_DIR}/hadoop/yarn/sbin/* "${HADOOP_HOME}/sbin/"
  chown "${HADOOP_SETUP_USER}":yarn "${HADOOP_HOME}"/sbin/*

  if [ ! -d "$YARN_GC_LOG_DIR" ]; then
    mkdir -p "$YARN_GC_LOG_DIR"
    chown "${HADOOP_SETUP_USER}":yarn "${YARN_GC_LOG_DIR}"
    chmod 775 "${YARN_GC_LOG_DIR}"
  fi

  if [ ! -d "$YARN_LOG_DIR" ]; then
    mkdir -p "$YARN_LOG_DIR"
    chown "${HADOOP_SETUP_USER}":yarn "${YARN_LOG_DIR}"
    chmod 775 "${YARN_LOG_DIR}"
  fi

  if [ ! -d "$YARN_PID_DIR" ]; then
    mkdir -p "$YARN_PID_DIR"
    chown "${HADOOP_SETUP_USER}":yarn "${YARN_PID_DIR}"
    chmod 775 "${YARN_PID_DIR}"
  fi  
 
  sed -i "s/YARN_LOG_DIR_REPLACE/${escape_yarn_log_dir}/g" "${INSTALL_TEMP_DIR}/hadoop/yarn/etc/hadoop/hadoop-env.sh"
  sed -i "s/YARN_PID_DIR_REPLACE/${escape_yarn_pid_dir}/g" "${INSTALL_TEMP_DIR}/hadoop/yarn/etc/hadoop/hadoop-env.sh" 
  sed -i "s/JAVA_HOME_REPLACE/${escape_java_home}/g" "${INSTALL_TEMP_DIR}/hadoop/yarn/etc/hadoop/hadoop-env.sh"
  sed -i "s/HADOOP_HOME_REPLACE/${escape_hadoop_home}/g" "${INSTALL_TEMP_DIR}/hadoop/yarn/etc/hadoop/hadoop-env.sh"
  sed -i "s/GC_LOG_DIR_REPLACE/${escape_yarn_gc_log_dir}/g" "${INSTALL_TEMP_DIR}/hadoop/yarn/etc/hadoop/hadoop-env.sh"
  cp -R "${INSTALL_TEMP_DIR}/hadoop/yarn/etc/hadoop/hadoop-env.sh" "${HADOOP_HOME}/etc/hadoop/"

  sed -i "s/GC_LOG_DIR_REPLACE/${escape_yarn_gc_log_dir}/g" "${INSTALL_TEMP_DIR}/hadoop/yarn/etc/hadoop/yarn-env.sh"
  cp -R "${INSTALL_TEMP_DIR}/hadoop/yarn/etc/hadoop/yarn-env.sh" "${HADOOP_HOME}/etc/hadoop/"

  sed -i "s/GC_LOG_DIR_REPLACE/${escape_yarn_gc_log_dir}/g" "${INSTALL_TEMP_DIR}/hadoop/yarn/etc/hadoop/mapred-env.sh"
  cp -R "${INSTALL_TEMP_DIR}/hadoop/yarn/etc/hadoop/mapred-env.sh" "${HADOOP_HOME}/etc/hadoop/"

cat<<HELPINFO
You can use the start/stop script in the ${HADOOP_HOME}/sbin/ directory to start or stop the various services of the yarn.
HELPINFO
}

function install_yarn_service_insecure() {
cat<<HELPINFO
You also need to set the yarn user to be a proxyable user, 
otherwise you will not be able to get the status of the service. 
Modify method: In core-site.xml, add parameters:
-----------------------------------------------------------------
<property>
<name>hadoop.proxyuser.yarn.hosts</name>
<value>*</value>
</property>
<property>
<name>hadoop.proxyuser.yarn.groups</name>
<value>*</value>
</property>
-----------------------------------------------------------------
HELPINFO
}

function install_registery_dns_insecure() {
  sed -i "s/YARN_REGISTRY_DNS_HOST_REPLACE/${YARN_REGISTRY_DNS_HOST}/g" "$INSTALL_TEMP_DIR/hadoop/yarn/etc/hadoop/yarn-site.xml"
  sed -i "s/YARN_REGISTRY_DNS_HOST_PORT_REPLACE/${YARN_REGISTRY_DNS_HOST_PORT}/g" "$INSTALL_TEMP_DIR/hadoop/yarn/etc/hadoop/yarn-site.xml"
}

function install_job_history_insecure() {
  sed -i "s/YARN_JOB_HISTORY_HOST_REPLACE/${YARN_JOB_HISTORY_HOST}/g" "$INSTALL_TEMP_DIR/hadoop/yarn/etc/hadoop/mapred-site.xml"
}

## @description  install yarn timeline server
## @audience     public
## @stability    stable
## http://hadoop.apache.org/docs/r3.1.0/hadoop-yarn/hadoop-yarn-site/TimelineServer.html
function install_timeline_server_insecure()
{
  find="/"
  replace="\\/"
  escape_aggregated_log_dir=${YARN_AGGREGATED_LOG_DIR//$find/$replace}
  escape_yarn_timeline_service_hbase_configuration_file=${YARN_TIMELINE_SERVICE_HBASE_CONFIGURATION_FILE//$find/$replace}
  escape_yarn_timeline_fs_store_dir=${YARN_TIMELINE_FS_STORE_DIR//$find/$replace}
  # timeline v1.5
  escape_yarn_timeline_service_leveldb_state_store_path=${YARN_TIMELINE_SERVICE_LEVELDB_STATE_STORE_PATH//$find/$replace}

  # set leveldb configuration
  sed -i "s/YARN_AGGREGATED_LOG_DIR_REPLACE/${escape_aggregated_log_dir}/g" "$INSTALL_TEMP_DIR/hadoop/yarn/etc/hadoop/yarn-site.xml"
  sed -i "s/YARN_TIMELINE_SERVICE_HBASE_CONFIGURATION_FILE_REPLACE/${escape_yarn_timeline_service_hbase_configuration_file}/g" "$INSTALL_TEMP_DIR/hadoop/yarn/etc/hadoop/yarn-site.xml"
  # timeline v1.5
  sed -i "s/YARN_TIMELINE_HOST_REPLACE/${YARN_TIMELINE_HOST}/g" "$INSTALL_TEMP_DIR/hadoop/yarn/etc/hadoop/yarn-site.xml"
  sed -i "s/YARN_TIMELINE_SERVICE_LEVELDB_STATE_STORE_PATH_REPLACE/${escape_yarn_timeline_service_leveldb_state_store_path}/g" "$INSTALL_TEMP_DIR/hadoop/yarn/etc/hadoop/yarn-site.xml"
  sed -i "s/YARN_TIMELINE_FS_STORE_DIR_REPLACE/${escape_yarn_timeline_fs_store_dir}/g" "$INSTALL_TEMP_DIR/hadoop/yarn/etc/hadoop/yarn-site.xml"

  host=$(hostname)
  if [ "x$YARN_TIMELINE_HOST" != "x$host" ]; then
    return 0
  fi

  echo -n "Do you want to create hdfs directories for timelineserver[Y/N]?"
  read -r answer
  echo "$answer"
  if [[ "$answer" = "y" || "$answer" = "Y" ]]; then
    echo "Continue installing ..."
  else
    echo "Stop creating hdfs directories for timelineserver"
    return 0
  fi

  echo "install yarn timeline server V1.5 ..."

cat<<HELPINFO
Create "${YARN_AGGREGATED_LOG_DIR}, ${YARN_TIMELINE_FS_STORE_DIR}"
path on hdfs, Owner is 'yarn', group is 'hadoop', 
and 'hadoop' group needs to include 'hdfs, yarn, mapred' yarn-site.xml users, etc.
HELPINFO

  if [[ ! -d "${YARN_TIMELINE_SERVICE_LEVELDB_STATE_STORE_PATH}" ]]; then
    mkdir -p "${YARN_TIMELINE_SERVICE_LEVELDB_STATE_STORE_PATH}"
    chown yarn "${YARN_TIMELINE_SERVICE_LEVELDB_STATE_STORE_PATH}"
  fi

  pathExists=$(pathExitsOnHDFS "${YARN_AGGREGATED_LOG_DIR}")
  if [[ -z "${pathExists}" ]]; then
    "${HADOOP_HOME}/bin/hadoop" dfs -mkdir -p "${YARN_AGGREGATED_LOG_DIR}"
    "${HADOOP_HOME}/bin/hadoop" dfs -chown yarn:hadoop "${YARN_AGGREGATED_LOG_DIR}"
    "${HADOOP_HOME}/bin/hadoop" dfs -chmod 1777 "${YARN_AGGREGATED_LOG_DIR}"
  fi

  pathExists=$(pathExitsOnHDFS "${YARN_TIMELINE_FS_STORE_DIR}")
  if [[ -z "${pathExists}" ]]; then
    # yarn.timeline-service.entity-group-fs-store.active-dir in yarn-site.xml
    "${HADOOP_HOME}/bin/hadoop" dfs -mkdir -p "${YARN_TIMELINE_FS_STORE_DIR}"
    "${HADOOP_HOME}/bin/hadoop" dfs -chown yarn:hadoop "${YARN_TIMELINE_FS_STORE_DIR}"
  fi

  pathExists=$(pathExitsOnHDFS "${YARN_TIMELINE_FS_STORE_DIR}/active")
  if [[ -z "${pathExists}" ]]; then
    # yarn.timeline-service.entity-group-fs-store.active-dir in yarn-site.xml
    "${HADOOP_HOME}/bin/hadoop" dfs -mkdir -p "${YARN_TIMELINE_FS_STORE_DIR}/active"
    "${HADOOP_HOME}/bin/hadoop" dfs -chown yarn:hadoop "${YARN_TIMELINE_FS_STORE_DIR}/active"
    "${HADOOP_HOME}/bin/hadoop" dfs -chmod 1777 "${YARN_TIMELINE_FS_STORE_DIR}/active"
  fi

  pathExists=$(pathExitsOnHDFS "${YARN_TIMELINE_FS_STORE_DIR}/done")
  if [[ -z "${pathExists}" ]]; then
    # yarn.timeline-service.entity-group-fs-store.done-dir in yarn-site.xml
    "${HADOOP_HOME}/bin/hadoop" dfs -mkdir -p "${YARN_TIMELINE_FS_STORE_DIR}/done"
    "${HADOOP_HOME}/bin/hadoop" dfs -chown yarn:hadoop "${YARN_TIMELINE_FS_STORE_DIR}/done"
    "${HADOOP_HOME}/bin/hadoop" dfs -chmod 0700 "${YARN_TIMELINE_FS_STORE_DIR}/done"
  fi
  
  ## install yarn timeline server V2
  echo "install yarn timeline server V2 ..."

cat<<HELPINFO
1. Use the hbase shell as the hbase user to authorize the yarn, HTTP user:
> grant 'yarn', 'RWC'
> grant 'HTTP', 'R'
HELPINFO
  echo -n "Have you done the above operation[Y/N]?"
  read -r answer
  if [[ "$answer" = "y" || "$answer" = "Y" ]]; then
    echo "Continue installing ..."
  else
    echo "Stop installing the timeline server V2"
    return 0
  fi
  
  pathExists=$(pathExitsOnHDFS "/hbase")
  if [[ -z "${pathExists}" ]]; then
    "${HADOOP_HOME}/bin/hadoop" fs -mkdir -p "/hbase"
    "${HADOOP_HOME}/bin/hadoop" fs -chmod -R 755 "/hbase"
  fi

  pathExists=$(pathExitsOnHDFS "/hbase/coprocessor")
  if [[ -z "${pathExists}" ]]; then
    "${HADOOP_HOME}/bin/hadoop" fs -mkdir -p "/hbase/coprocessor"
    "${HADOOP_HOME}/bin/hadoop" fs -chmod -R 755 "/hbase/coprocessor"
  fi

  "${HADOOP_HOME}/bin/hadoop" fs -put "${HADOOP_HOME}"/share/hadoop/yarn/timelineservice/hadoop-yarn-server-timelineservice-hbase-coprocessor-3.*.jar "/hbase/coprocessor/hadoop-yarn-server-timelineservice.jar"
  "${HADOOP_HOME}/bin/hadoop" fs -chmod 755 "/hbase/coprocessor/hadoop-yarn-server-timelineservice.jar"

cat<<HELPINFO
2. Copy the timeline hbase jar to the <hbase_client>/lib path:
HELPINFO

  if [[ -n "${HBASE_HOME}" ]]; then
    cp "${HADOOP_HOME}"/share/hadoop/yarn/timelineservice/hadoop-yarn-server-timelineservice-hbase-common-3.*-SNAPSHOT.jar "${HBASE_HOME}/lib"
    cp "${HADOOP_HOME}"/share/hadoop/yarn/timelineservice/hadoop-yarn-server-timelineservice-hbase-client-3.*-SNAPSHOT.jar "${HBASE_HOME}/lib"
    cp "${HADOOP_HOME}"/share/hadoop/yarn/timelineservice/hadoop-yarn-server-timelineservice-3.*-SNAPSHOT.jar "${HBASE_HOME}/lib"
  fi

cat<<HELPINFO
3. In the hbase server, After using the keytab authentication of yarn, 
In the <hbase_client> path, Execute the following command to create a schema
> bin/hbase org.apache.hadoop.yarn.server.timelineservice.storage.TimelineSchemaCreator -create
HELPINFO
  echo -n "Have you done the above operation[Y/N]?"
  read -r answer
  if [[ "$answer" = "y" || "$answer" = "Y" ]]; then
    echo "Continue installing ..."
  else
    echo "Please initialize hbase timeline schema before you start timelineserver"
  fi
}

## @description  start yarn
## @audience     public
## @stability    stable
function start_yarn()
{
  current_user=$(whoami)
  host=$(hostname)
  
  # Start RM if the host is in YARN_RESOURCE_MANAGER_HOSTS
  index=$(indexByRMHosts "${host}")
  if [ -n "$index" ]; then
    # Only RM needs to execute the following code
    echo "Starting resourcemanager..."
    if [ ${current_user} != "yarn" ]; then
      su - yarn -c "${HADOOP_HOME}/sbin/start-resourcemanager.sh"
    else
      "${HADOOP_HOME}/sbin/start-resourcemanager.sh"
    fi
  fi

  # Start nodemanager
  index=$(indexOfNMExcludeHosts "${host}")
  if [ -z "$index" ]; then
    echo "Starting nodemanager..."
    if [ ${current_user} != "yarn" ]; then
      su - yarn -c "${HADOOP_HOME}/sbin/start-nodemanager.sh"
    else
      "${HADOOP_HOME}/sbin/start-nodemanager.sh"
    fi
  fi

  # Start timeline
  if [ "x$YARN_TIMELINE_HOST" = "x$host" ]; then
    echo "Starting timelineserver..."
    if [ ${current_user} != "yarn" ]; then
      su - yarn -c "${HADOOP_HOME}/sbin/start-timelineserver.sh"
      su - yarn -c "${HADOOP_HOME}/sbin/start-timelinereader.sh"
    else
      ${HADOOP_HOME}/sbin/start-timelineserver.sh
      ${HADOOP_HOME}/sbin/start-timelinereader.sh
    fi
  fi

  # Start jobhistory
  if [ "x$YARN_JOB_HISTORY_HOST" = "x$host" ]; then
    echo "Starting mapreduce job history..."
    if [ ${current_user} != "yarn" ]; then
      su - yarn -c "${HADOOP_HOME}/sbin/start-mr-jobhistory.sh"
    else
      ${HADOOP_HOME}/sbin/start-mr-jobhistory.sh
    fi
  fi

  # Start registrydns
  if [ "x$YARN_REGISTRY_DNS_HOST" = "x$host" ]; then
    echo "Starting registry dns..."
    sudo ${HADOOP_HOME}/sbin/start-registrydns.sh
  fi
}

## @description  stop yarn
## @audience     public
## @stability    stable
function stop_yarn()
{
  current_user=$(whoami)
  host=$(hostname)
  # Stop RM if the host is in YARN_RESOURCE_MANAGER_HOSTS
  index=$(indexByRMHosts "${host}")
  if [ -n "$index" ]; then
    # Only RM needs to execute the following code
    if [ ${current_user} != "yarn" ]; then
      echo "Stopping resourcemanager..."
      su - yarn -c "${HADOOP_HOME}/sbin/stop-resourcemanager.sh"
    else
      "${HADOOP_HOME}/sbin/stop-resourcemanager.sh"
    fi
  fi

  # Stop nodemanager
  index=$(indexOfNMExcludeHosts "${host}")
  if [ -z "$index" ]; then
    echo "Stopping nodemanager..."
    if [ ${current_user} != "yarn" ]; then
      su - yarn -c "${HADOOP_HOME}/sbin/stop-nodemanager.sh"
    else
      "${HADOOP_HOME}/sbin/stop-nodemanager.sh"
    fi
  fi  

  # Stop timeline
  if [ "x$YARN_TIMELINE_HOST" = "x$host" ]; then
    echo "Stopping timelineserver..."
    if [ ${current_user} != "yarn" ]; then
      su - yarn -c "${HADOOP_HOME}/sbin/stop-timelineserver.sh"
      su - yarn -c "${HADOOP_HOME}/sbin/stop-timelinereader.sh"
    else
      ${HADOOP_HOME}/sbin/stop-timelineserver.sh
      ${HADOOP_HOME}/sbin/stop-timelinereader.sh
    fi
  fi

  # Stop jobhistory
  if [ "x$YARN_JOB_HISTORY_HOST" = "x$host" ]; then
    echo "Stopping mapreduce job history..."
    if [ ${current_user} != "yarn" ]; then
      su - yarn -c "${HADOOP_HOME}/sbin/stop-mr-jobhistory.sh"
    else
      ${HADOOP_HOME}/sbin/stop-mr-jobhistory.sh
    fi
  fi

  # Stop registrydns
  if [ "x$YARN_REGISTRY_DNS_HOST" = "x$host" ]; then
    echo "Stopping registry dns..."
    sudo ${HADOOP_HOME}/sbin/stop-registrydns.sh
  fi
}

