<!--
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# mini-submarine

This is a docker image built for submarine development and quick start.

Please Note: don't use the image in production environment. It's only for test purpose.

### Build the image

#### Use the image we provide

> Tag 0.2.0 indicates the version number of hadoop submarine in images

```
docker push hadoopsubmarine/mini-submarine:0.2.0 
```

#### Create image by yourself

> you may need a VPN if your network is limited

```
./build_mini-submarine.sh
```

#### Run mini-submarine image

```
docker run -it -h submarine-dev --net=bridge --privileged local/hadoop-docker:submarine /bin/bash

# In the container, use root user to bootstrap hdfs and yarn
/tmp/hadoop-config/bootstrap.sh

# Two commands to check if yarn and hdfs is running as expected
yarn node -list -showDetails
```

##### You should see info like this:

```
Total Nodes:1
         Node-Id      Node-State	Node-Http-Address	Number-of-Running-Containers
submarine-dev:35949         RUNNING	submarine-dev:8042                            0
Detailed Node Information :
  Configured Resources : <memory:8192, vCores:16, nvidia.com/gpu: 1>
  Allocated Resources : <memory:0, vCores:0>
  Resource Utilization by Node : PMem:4144 MB, VMem:4189 MB, VCores:0.25308025
  Resource Utilization by Containers : PMem:0 MB, VMem:0 MB, VCores:0.0
  Node-Labels :
```

```
hdfs dfs -ls /user
```

> drwxr-xr-x   - yarn supergroup          0 2019-07-22 07:59 /user/yarn



### Use yarn user to run job

```
su yarn
```

### To run a mnist TF job with submarine + TonY runtime
```
cd && cd submarine && ./run_submarine_minist_tony.sh
```

### To try your own submarine program

run container with your source code. You can also use "docker cp" to an existing running container

1. `docker run -it -h submarine-dev --net=bridge --privileged -v pathToMyScrit.py:/home/yarn/submarine/myScript.py local/hadoop-docker:submarine /bin/bash`

2. refer to the `run_submarine_minist_tony.sh` and modify the script to your script

3. Try to run it. Since this is a single node environment, keep in mind that the workers could have conflicts with each other. For instance, the mnist_distributed.py example has a workaround to fix the conflicts when two workers are using same "data_dir" to download data set.


### To run a DistributedShell job with Docker
```
cd && ./yarn-ds-docker.sh
```

### To run a spark job
```
cd && cd spark-script && ./run_spark.sh
```
