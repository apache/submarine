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


###mod
export YARN_ROOT_LOGGER=INFO,DRFA
#export YARN_ROOT_LOGGER=DEBUG,DRFA
export YARN_LOG_DIR=$HADOOP_LOG_DIR
export YARN_PID_DIR=$HADOOP_PID_DIR

GC_LOG_DATE=`date +%F`
GC_LOG_DIR="GC_LOG_DIR_REPLACE"
#export RESOURCEMANAGER_JVM_OPTS="-Dcom.netease.appname=ResourceManager -Djava.net.preferIPv4Stack=true -verbose:gc -XX:+PrintGCDetails -XX:+PrintGCDateStamps -XX:+PrintTenuringDistribution -Xloggc:$GC_LOG_DIR/resourcemanager.gc.log.$GC_LOG_DATE -XX:+UseGCLogFileRotation -XX:NumberOfGCLogFiles=3 -XX:GCLogFileSize=512M -server -Xmx20480m -Xms20480m -Xmn5120m -XX:SurvivorRatio=4 -Xss256k -XX:PermSize=256m -XX:MaxPermSize=256m -XX:+UseParNewGC -XX:MaxTenuringThreshold=15 -XX:+UseConcMarkSweepGC -XX:+CMSParallelRemarkEnabled -XX:+UseCMSCompactAtFullCollection -XX:+CMSClassUnloadingEnabled -XX:+UseCMSInitiatingOccupancyOnly -XX:CMSInitiatingOccupancyFraction=75 -XX:-DisableExplicitGC"
export RESOURCEMANAGER_JVM_OPTS="-Dcom.netease.appname=ResourceManager -Djava.net.preferIPv4Stack=true -verbose:gc -XX:+PrintGCDetails -XX:+PrintGCDateStamps -XX:+PrintTenuringDistribution -Xloggc:$GC_LOG_DIR/resourcemanager.gc.log.$GC_LOG_DATE -XX:+UseGCLogFileRotation -XX:NumberOfGCLogFiles=3 -Xdebug -Xrunjdwp:transport=dt_socket,server=y,suspend=n,address=6001"
export NODEMANAGER_JVM_OPTS="-Dcom.netease.appname=NodeManager -Djava.net.preferIPv4Stack=true -verbose:gc -XX:+PrintGCDetails -XX:+PrintGCDateStamps -XX:+PrintTenuringDistribution -Xloggc:$GC_LOG_DIR/nodemanager.gc.log.$GC_LOG_DATE -XX:+UseGCLogFileRotation -XX:NumberOfGCLogFiles=3 -XX:GCLogFileSize=512M -server -Xmx5120m -Xms5120m -Xmn1024m -XX:SurvivorRatio=4 -Xss256k -XX:PermSize=256m -XX:MaxPermSize=256m -XX:+UseParNewGC -XX:MaxTenuringThreshold=15 -XX:+UseConcMarkSweepGC -XX:+CMSParallelRemarkEnabled -XX:+UseCMSCompactAtFullCollection -XX:+CMSClassUnloadingEnabled -XX:+UseCMSInitiatingOccupancyOnly -XX:CMSInitiatingOccupancyFraction=75 -XX:-DisableExplicitGC"
#export NODEMANAGER_JVM_OPTS="-Dcom.netease.appname=NodeManager -Djava.net.preferIPv4Stack=true -verbose:gc -XX:+PrintGCDetails -XX:+PrintGCDateStamps -XX:+PrintTenuringDistribution -Xloggc:$GC_LOG_DIR/nodemanager.gc.log.$GC_LOG_DATE -XX:+UseGCLogFileRotation -XX:NumberOfGCLogFiles=3  -Xdebug -Xrunjdwp:transport=dt_socket,server=y,suspend=y,address=6001"
###mod
#export YARN_TIMELINESERVER_OPTS="-Dcom.netease.appname=TimelineServer -Djava.net.preferIPv4Stack=true -verbose:gc -XX:+PrintGCDetails -XX:+PrintGCDateStamps -XX:+PrintTenuringDistribution -Xloggc:$GC_LOG_DIR/timelineserver.gc.log.$GC_LOG_DATE -XX:+UseGCLogFileRotation -XX:NumberOfGCLogFiles=3 -Xdebug -Xrunjdwp:transport=dt_socket,server=y,suspend=n,address=6001"
#export YARN_TIMELINEREADER_OPTS="-Dcom.netease.appname=TimelineReaderServer -Djava.net.preferIPv4Stack=true -verbose:gc -XX:+PrintGCDetails -XX:+PrintGCDateStamps -XX:+PrintTenuringDistribution -Xloggc:$GC_LOG_DIR/timelineserver.gc.log.$GC_LOG_DATE -XX:+UseGCLogFileRotation -XX:NumberOfGCLogFiles=3 -Xdebug -Xrunjdwp:transport=dt_socket,server=y,suspend=y,address=6001"

# User for YARN daemons
export HADOOP_YARN_USER=${HADOOP_YARN_USER:-yarn}

# resolve links - $0 may be a softlink
export YARN_CONF_DIR="${YARN_CONF_DIR:-$HADOOP_YARN_HOME/etc/hadoop}"

# some Java parameters
# export JAVA_HOME=/home/y/libexec/jdk1.6.0/
if [ "$JAVA_HOME" != "" ]; then
  #echo "run java in $JAVA_HOME"
  JAVA_HOME=$JAVA_HOME
fi
  
if [ "$JAVA_HOME" = "" ]; then
  echo "Error: JAVA_HOME is not set."
  exit 1
fi

JAVA=$JAVA_HOME/bin/java
JAVA_HEAP_MAX=-Xmx1000m 

# For setting YARN specific HEAP sizes please use this
# Parameter and set appropriately
# YARN_HEAPSIZE=1000

# check envvars which might override default args
if [ "$YARN_HEAPSIZE" != "" ]; then
  JAVA_HEAP_MAX="-Xmx""$YARN_HEAPSIZE""m"
fi

# Resource Manager specific parameters

# Specify the max Heapsize for the ResourceManager using a numerical value
# in the scale of MB. For example, to specify an jvm option of -Xmx1000m, set
# the value to 1000.
# This value will be overridden by an Xmx setting specified in either YARN_OPTS
# and/or YARN_RESOURCEMANAGER_OPTS.
# If not specified, the default value will be picked from either YARN_HEAPMAX
# or JAVA_HEAP_MAX with YARN_HEAPMAX as the preferred option of the two.
#export YARN_RESOURCEMANAGER_HEAPSIZE=1000

# Specify the JVM options to be used when starting the ResourceManager.
# These options will be appended to the options specified as YARN_OPTS
# and therefore may override any similar flags set in YARN_OPTS
export YARN_RESOURCEMANAGER_OPTS="$RESOURCEMANAGER_JVM_OPTS"

# Node Manager specific parameters

# Specify the max Heapsize for the NodeManager using a numerical value
# in the scale of MB. For example, to specify an jvm option of -Xmx1000m, set
# the value to 1000.
# This value will be overridden by an Xmx setting specified in either YARN_OPTS
# and/or YARN_NODEMANAGER_OPTS.
# If not specified, the default value will be picked from either YARN_HEAPMAX
# or JAVA_HEAP_MAX with YARN_HEAPMAX as the preferred option of the two.
#export YARN_NODEMANAGER_HEAPSIZE=1000

# Specify the JVM options to be used when starting the NodeManager.
# These options will be appended to the options specified as YARN_OPTS
# and therefore may override any similar flags set in YARN_OPTS
export YARN_NODEMANAGER_OPTS="$NODEMANAGER_JVM_OPTS"

# so that filenames w/ spaces are handled correctly in loops below
IFS=


# default log directory & file
if [ "$YARN_LOG_DIR" = "" ]; then
  YARN_LOG_DIR="$HADOOP_YARN_HOME/logs"
fi
if [ "$YARN_LOGFILE" = "" ]; then
  YARN_LOGFILE='yarn.log'
fi

# default policy file for service-level authorization
if [ "$YARN_POLICYFILE" = "" ]; then
  YARN_POLICYFILE="hadoop-policy.xml"
fi

# restore ordinary behaviour
unset IFS


YARN_OPTS="$YARN_OPTS -Dhadoop.log.dir=$YARN_LOG_DIR"
YARN_OPTS="$YARN_OPTS -Dyarn.log.dir=$YARN_LOG_DIR"
YARN_OPTS="$YARN_OPTS -Dhadoop.log.file=$YARN_LOGFILE"
YARN_OPTS="$YARN_OPTS -Dyarn.log.file=$YARN_LOGFILE"
YARN_OPTS="$YARN_OPTS -Dyarn.home.dir=$YARN_COMMON_HOME"
YARN_OPTS="$YARN_OPTS -Dyarn.id.str=$YARN_IDENT_STRING"
YARN_OPTS="$YARN_OPTS -Dhadoop.root.logger=${YARN_ROOT_LOGGER:-INFO,console}"
YARN_OPTS="$YARN_OPTS -Dyarn.root.logger=${YARN_ROOT_LOGGER:-INFO,console}"
if [ "x$JAVA_LIBRARY_PATH" != "x" ]; then
  YARN_OPTS="$YARN_OPTS -Djava.library.path=$JAVA_LIBRARY_PATH"
fi  
YARN_OPTS="$YARN_OPTS -Dyarn.policy.file=$YARN_POLICYFILE"
