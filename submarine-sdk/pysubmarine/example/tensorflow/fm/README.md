<!---
  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License. See accompanying LICENSE file.
-->
### Running FM on a local machine
1. Create a JSON configuration file containing train,valid and test data, model parameters, 
metrics, save model path, resources. e.g. [fm.json](./fm.json)

2. Train
```
python run_fm.py -conf=fm.json -task_type train
```
3. Evaluate
```
python run_fm.py -conf=fm.json -task_type evaluate
```
### Running FM on Submarine
1. Upload data to a shared file system like hdfs, s3.

2. Create a JSON configuration file for distributed training. e.g. [fm_distributed.json](./fm_distributed.json)

3. Submit Job
```
SUBMARINE_VERSION=0.5.0
SUBMARINE_HADOOP_VERSION=2.9

java -cp $(${HADOOP_COMMON_HOME}/bin/hadoop classpath --glob):submarine-all-${SUBMARINE_VERSION}-hadoop-${SUBMARINE_HADOOP_VERSION}.jar:${HADOOP_CONF_PATH} \
 org.apache.submarine.client.cli.Cli job run --name fm-job-001 \
 --framework tensorflow \
 --verbose \
 --input_path "" \
 --num_workers 2 \
 --worker_resources memory=4G,vcores=4 \
 --num_ps 1 \
 --ps_resources memory=4G,vcores=4 \
 --worker_launch_cmd "myvenv.zip/venv/bin/python run_fm.py -conf=fm_distributed.json" \
 --ps_launch_cmd "myvenv.zip/venv/bin/python run_fm.py -conf=fm_distributed.json" \
 --insecure \
 --conf tony.containers.resources=myvenv.zip#archive,submarine-all-${SUBMARINE_VERSION}-hadoop-${SUBMARINE_HADOOP_VERSION}.jar,fm_distributed.json,run_fm.py
```
