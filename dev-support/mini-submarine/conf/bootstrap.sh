#!/bin/bash
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# start dockerd
nohup dockerd --host=unix:///var/run/docker.sock > /var/log/dockerd.log 2>&1 &

: ${HADOOP_PREFIX:=/usr/local/hadoop}

. $HADOOP_PREFIX/etc/hadoop/hadoop-env.sh

# Directory to find config artifacts
CONFIG_DIR="/tmp/hadoop-config"


# Copy config files from volume mount

for f in slaves core-site.xml hdfs-site.xml mapred-site.xml yarn-site.xml container-executor.cfg capacity-scheduler.xml node-resources.xml resource-types.xml submarine-site.xml; do
  if [[ -e ${CONFIG_DIR}/$f ]]; then
    cp ${CONFIG_DIR}/$f $HADOOP_PREFIX/etc/hadoop/$f
  else
    echo "ERROR: Could not find $f in $CONFIG_DIR"
    exit 1
  fi
done

# create cgroups
mkdir -p /sys/fs/cgroup/cpu/hadoop-yarn
chown -R yarn /sys/fs/cgroup/cpu/hadoop-yarn
mkdir -p /sys/fs/cgroup/memory/hadoop-yarn
chown -R yarn /sys/fs/cgroup/memory/hadoop-yarn
mkdir -p /sys/fs/cgroup/blkio/hadoop-yarn
chown -R yarn /sys/fs/cgroup/blkio/hadoop-yarn
mkdir -p /sys/fs/cgroup/net_cls/hadoop-yarn
chown -R yarn /sys/fs/cgroup/net_cls/hadoop-yarn
mkdir -p /sys/fs/cgroup/devices/hadoop-yarn
chown -R yarn /sys/fs/cgroup/devices/hadoop-yarn

# set container-executor permission
chmod 6050 /usr/local/hadoop/bin/container-executor
chmod 0400 /usr/local/hadoop/etc/hadoop/container-executor.cfg

# creat log and app dir
mkdir ${HADOOP_PREFIX}/logs
chown yarn:hadoop ${HADOOP_PREFIX}/logs
mkdir /var/lib/hadoop-yarn
chown yarn:hadoop /var/lib/hadoop-yarn
mkdir /var/log/hadoop-yarn
chown yarn:hadoop /var/log/hadoop-yarn
# installing libraries if any - (resource urls added comma separated to the ACP system variable)
cd $HADOOP_PREFIX/share/hadoop/common ; for cp in ${ACP//,/ }; do  echo == $cp; curl -LO $cp ; done; cd -

if [[ "${HOSTNAME}" =~ "submarine-dev" ]]; then
  mkdir -p /root/hdfs/namenode
  $HADOOP_PREFIX/bin/hdfs namenode -format -force -nonInteractive
  sed -i s/hdfs-nn/0.0.0.0/ /usr/local/hadoop/etc/hadoop/core-site.xml
  nohup $HADOOP_PREFIX/sbin/hadoop-daemon.sh start namenode > /tmp/nn.log 2>&1 &
fi

if [[ "${HOSTNAME}" =~ "submarine-dev" ]]; then
  mkdir -p /root/hdfs/datanode
  #  wait up to 30 seconds for namenode
 # count=0 && while [[ $count -lt 15 && -z `curl -sf http://submarine-dev:50070` ]]; do echo "Waiting for hdfs-nn" ; ((count=count+1)) ; sleep 2; done
  #[[ $count -eq 15 ]] && echo "Timeout waiting for hdfs-nn, exiting." && exit 1
  sleep 5
  nohup $HADOOP_PREFIX/sbin/hadoop-daemon.sh start datanode > /tmp/dn.log 2>&1 &
fi

#add yarn to hdfs user group, create hdfs folder and export hadoop path
groupadd supergroup
usermod -aG supergroup yarn
usermod -aG docker yarn
su yarn -c "/usr/local/hadoop/bin/hadoop fs -mkdir -p /user/yarn"
echo "export PATH=$PATH:/usr/local/hadoop/bin" >> /home/yarn/.bashrc

if [[ "${HOSTNAME}" =~ "submarine-dev" ]]; then
  sed -i s/yarn-rm/0.0.0.0/ $HADOOP_PREFIX/etc/hadoop/yarn-site.xml

  # start zk
  su yarn -c "cd /usr/local/zookeeper && /usr/local/zookeeper/bin/zkServer.sh start >/dev/null 2>&1"

  cp ${CONFIG_DIR}/start-yarn-rm.sh $HADOOP_PREFIX/sbin/
  cd $HADOOP_PREFIX/sbin
  chmod +x start-yarn-rm.sh
  su yarn -c "/usr/local/hadoop/sbin/start-yarn-rm.sh"
fi

if [[ "${HOSTNAME}" =~ "submarine-dev" ]]; then
  sed -i '/<\/configuration>/d' $HADOOP_PREFIX/etc/hadoop/yarn-site.xml
  cat >> $HADOOP_PREFIX/etc/hadoop/yarn-site.xml <<- EOM
  <property>
    <name>yarn.nodemanager.resource.memory-mb</name>
    <value>8192</value>
  </property>

  <property>
    <name>yarn.nodemanager.resource.cpu-vcores</name>
    <value>16</value>
  </property>
EOM
  echo '</configuration>' >> $HADOOP_PREFIX/etc/hadoop/yarn-site.xml
  cp ${CONFIG_DIR}/start-yarn-nm.sh $HADOOP_PREFIX/sbin/
  cd $HADOOP_PREFIX/sbin
  chmod +x start-yarn-nm.sh
  
  #copy the distributed shell script for debug purpose
  cp ${CONFIG_DIR}/yarn-ds-docker.sh /home/yarn/
  chown yarn /home/yarn/yarn-ds-docker.sh
  chmod +x /home/yarn/yarn-ds-docker.sh
  
  #  wait up to 30 seconds for resourcemanager
  count=0 && while [[ $count -lt 30 && -z `curl -sf http://submarine-dev:8088/ws/v1/cluster/info` ]]; do echo "Waiting for yarn-rm" ; ((count=count+1)) ; sleep 1; done
  [[ $count -eq 30 ]] && echo "Timeout waiting for yarn-rm, exiting." && exit 1

  su yarn -c "/usr/local/hadoop/sbin/start-yarn-nm.sh"
fi

if [[ $1 == "-d" ]]; then
  until find ${HADOOP_PREFIX}/logs -mmin -1 | egrep -q '.*'; echo "`date`: Waiting for logs..." ; do sleep 2 ; done
  tail -F ${HADOOP_PREFIX}/logs/* &
  while true; do sleep 1000; done
fi

if [[ $1 == "-bash" ]]; then
  /bin/bash
fi
