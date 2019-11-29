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

# Execute command format : sge_run.sh %num_workers% %input_path%
if [ $# -ne 2 ] 
then
    echo "Parameter setting error!"
    exit 1
fi

WORKER_NUMBER=$1
CFG_PATH=$2

HOST_NAME=$(hostname)
JOB_NAME=`echo $HOST_NAME |awk -F "." '{print($2)}'`
CLUSTER_USER=`echo $HOST_NAME |awk -F "." '{print($3)}'`
DOMAIN_SUFFIX=`echo $HOST_NAME |awk -F "." '{for(i=4;i<=NF;i++){if(tostr == ""){tostr=$i}else{tostr=tostr"."$i};if(i==NF)print tostr}}'`

MASTER_HOST=""
WORKER_HOST_STRS=""
declare -a WORKER_HOST_LIST
SLOTS="30"

for ((num=0; num<$WORKER_NUMBER; num++))
do
   if [ $num -eq 0 ]
   then
       MASTER_HOST="master-0."$JOB_NAME"."$CLUSTER_USER"."$DOMAIN_SUFFIX
       WORKER_HOST_STRS="master-0."$JOB_NAME"."$CLUSTER_USER"."$DOMAIN_SUFFIX
       SLOTS=$SLOTS",[master-0."$JOB_NAME"."$CLUSTER_USER"."$DOMAIN_SUFFIX"=48]"
   else
       let tmpnum=$num-1
       WORKER_HOST_STRS=$WORKER_HOST_STRS",worker-"$tmpnum"."$JOB_NAME"."$CLUSTER_USER"."$DOMAIN_SUFFIX
       WORKER_HOST_LIST+=("worker-"$tmpnum"."$JOB_NAME"."$CLUSTER_USER"."$DOMAIN_SUFFIX)
       SLOTS=$SLOTS",[worker-"$tmpnum"."$JOB_NAME"."$CLUSTER_USER"."$DOMAIN_SUFFIX"=48]"
    fi
done

if [ $(echo $HOST_NAME |grep "^master-") ]
then
    sudo su - -c "sleep 30s && export DEBIAN_FRONTEND=noninteractive && apt-get update && \
    apt-get install -y gridengine-master gridengine-exec gridengine-client;apt-get autoremove -y && apt-get clean && \
    /etc/init.d/gridengine-master start && /etc/init.d/gridengine-exec start"

    sudo su - -s /bin/bash -c ". ${CFG_PATH}/gencfs.sh $WORKER_HOST_STRS $SLOTS"
    sudo su -c " qconf -Mc /tmp/qconf-mc.txt && qconf -Ae /tmp/qconf-ae.txt && qconf -as \`hostname\` && 
    qconf -Ap /tmp/qconf-ap.txt && qconf -Aq /tmp/qconf-aq.txt && qconf -am $CLUSTER_USER && 
    echo finish master >> ${CFG_PATH}/setcfg.log "
    for worker_num in ${WORKER_HOST_LIST[@]}
    do
        echo  add $worker_num
        sudo su -c " qconf -ah $worker_num && echo add worker node $worker_num >> ${CFG_PATH}/setcfg.log "
    done

elif [ $(echo $HOST_NAME |grep "^worker-") ]
then
    sudo su - -s /bin/bash -c "sleep 2m && echo please wait && echo please wait >> ${CFG_PATH}/setcfg.log"
    sudo su - -c "export DEBIAN_FRONTEND=noninteractive &&  apt-get update && \
    apt-get install -y gridengine-client gridengine-exec; apt-get autoremove -y && apt-get clean"
    sudo su - -c "echo $MASTER_HOST > /var/lib/gridengine/default/common/act_qmaster"
    sudo su - -c "/etc/init.d/gridengine-exec start && echo Start SGE for worker is finished >> ${CFG_PATH}/setcfg.log"

    sudo su - -s /bin/bash -c ". ${CFG_PATH}/gencfs.sh $WORKER_HOST_STRS $SLOTS"
    sudo su - -c "qconf -Me /tmp/qconf-ae.txt && echo done for $HOST_NAME worker. >> ${CFG_PATH}/setcfg.log"

else
    echo "hostname doesn't match! should start with master or worker!" 1>&2 exit 1
fi

#sudo su - -s /bin/bash -c "echo sge start to finish >> ${CFG_PATH}/setcfg.log"
