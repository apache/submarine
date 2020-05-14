export JAVA_HOME=${JAVA_HOME:-$HOME/workspace/app/java}
export HADOOP_HOME=${HADOOP_HOME:-$HADOOP_HDFS_HOME}
export CLASSPATH=${CLASSPATH:-`hdfs classpath --glob`}
export ARROW_LIBHDFS_DIR=${ARROW_LIBHDFS_DIR:-$HADOOP_HOME/lib/native}

# path to pysubmarine/submarine
PYTHONPATH=$HOME/workspace/submarine/submarine-sdk/pysubmarine

HADOOP_CONF_PATH=${HADOOP_CONF_PATH:-$HADOOP_CONF_DIR}

SUBMARINE_VERSION=0.4.0-SNAPSHOT
SUBMARINE_HADOOP_VERSION=2.9
SUBMARINE_JAR=/opt/submarine-dist-${SUBMARINE_VERSION}-hadoop-${SUBMARINE_HADOOP_VERSION}/submarine-dist-${SUBMARINE_VERSION}-hadoop-${SUBMARINE_HADOOP_VERSION}/submarine-all-${SUBMARINE_VERSION}-hadoop-${SUBMARINE_HADOOP_VERSION}.jar

java -cp $(${HADOOP_COMMON_HOME}/bin/hadoop classpath --glob):${SUBMARINE_JAR}:${HADOOP_CONF_PATH} \
 org.apache.submarine.client.cli.Cli job run --name deepfm-job-001 \
 --framework pytorch \
 --verbose \
 --input_path "" \
 --num_workers 2 \
 --worker_resources memory=1G,vcores=1 \
 --worker_launch_cmd "JAVA_HOME=$JAVA_HOME HADOOP_HOME=$HADOOP_HOME CLASSPATH=$CLASSPATH ARROW_LIBHDFS_DIR=$ARROW_LIBHDFS_DIR PYTHONPATH=$PYTHONPATH sdk.zip/sdk/bin/python run_ctr.py --conf ./deepfm.json --task_type train" \
 --insecure \
 --conf tony.containers.resources=sdk.zip#archive,${SUBMARINE_JAR},run_ctr.py,deepfm.json




