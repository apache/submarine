<!--
   Licensed to the Apache Software Foundation (ASF) under one or more
   contributor license agreements.  See the NOTICE file distributed with
   this work for additional information regarding copyright ownership.
   The ASF licenses this file to You under the Apache License, Version 2.0
   (the "License"); you may not use this file except in compliance with
   the License.  You may obtain a copy of the License at
   http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
-->

# Quick Start Guide

## Prerequisite

Must:

- Apache Hadoop version newer than 2.7.3

Optional:

- [Enable YARN Service](https://hadoop.apache.org/docs/r3.1.0/hadoop-yarn/hadoop-yarn-site/yarn-service/Overview.html) (Only yarn 3.1.x + needs)
- [Enable GPU on YARN 3.1.0+](https://hadoop.apache.org/docs/r3.1.0/hadoop-yarn/hadoop-yarn-site/UsingGpus.html)
- [Enable Docker on YARN 2.8.2+](https://hadoop.apache.org/docs/r2.8.2/hadoop-yarn/hadoop-yarn-site/DockerContainers.html)
- [Build Docker images](WriteDockerfileTF.md)


<br />

## Submarine Configuration

After submarine 0.2.0, it supports two runtimes which are YARN service runtime and Linkedin's TonY runtime for YARN. Each runtime can support both Tensorflow and PyTorch framework. But we don't need to worry about the usage because the two runtime implements the same interface.

So before we start running a job, the runtime type should be picked. The runtime choice may vary depending on different requirements. Check below table to choose your runtime.

Note that if you want to quickly try Submarine on new or existing YARN cluster, use TonY runtime will help you get start easier(0.2.0+)

| Hadoop (YARN) version | Is Docker Enabled | Is GPU Enabled | Acceptable Runtime  |
| :-------------------- | :---------------- | -------------- | ------------------- |
| 2.7.3 ～2.9.x         | X                 | X              | Tony Runtime        |
| 2.9.x ～3.1.0         | Y                 | Y              | Tony Runtime        |
| 3.1.0+                | Y                 | Y              | Tony / YANR-Service |

For the environment setup, please check [InstallationGuide](InstallationGuide.md) or [InstallationGuideCN](InstallationGuideChineseVersion.md)

Once the applicable runtime is chosen and environment is ready, a `submarine-site.xml` need to be created under  `$HADOOP_CONF_DIR`. To use the TonY runtime, please set below value in the submarine configuration.

|Configuration Name | Description |
|:---- |:---- |
| `submarine.runtime.class` | "org.apache.submarine.server.submitter.yarn.YarnRuntimeFactory" or "org.apache.submarine.server.submitter.yarnservice.YarnServiceRuntimeFactory" |

<br />

A sample `submarine-site.xml` is here:
```java
<?xml version="1.0"?>
<configuration>
  <property>
    <name>submarine.runtime.class</name>
    <value>org.apache.submarine.server.submitter.yarn.YarnRuntimeFactory</value>
    <!-- Alternatively, you can use:
    <value>org.apache.submarine.server.submitter.yarnservice.YarnServiceRuntimeFactory</value>
    -->
  </property>
</configuration>
```

For more Submarine configuration:

|Configuration Name | Description |
|:---- |:---- |
| `submarine.localization.max-allowed-file-size-mb` | Optional. This sets a size limit to the file/directory to be localized in "-localization" CLI option. 2GB by default. |


<br />

## Launch Training Job
This section will give us an idea of how the submarine CLI looks like.

Although the run job command looks simple, different job may have very different parameters.

For a quick try on Mnist example with TonY runtime, check [TonY Mnist Example](TonYRuntimeGuide.md)


For a quick try on Cifar10 example with YARN native service runtime, check [YARN Service Cifar10 Example](RunningDistributedCifar10TFJobs.md)


<br />
## Get job history / logs

### Get Job Status from CLI

```shell
CLASSPATH=path-to/hadoop-conf:path-to/hadoop-submarine-all-${SUBMARINE_VERSION}-hadoop-${HADOOP_VERSION}.jar \
java org.apache.submarine.client.cli.Cli job show --name tf-job-001
```

Output looks like:
```shell
Job Meta Info:
  Application Id: application_1532131617202_0005
  Input Path: hdfs://default/dataset/cifar-10-data
  Checkpoint Path: hdfs://default/tmp/cifar-10-jobdir
  Run Parameters: --name tf-job-001 --docker_image <your-docker-image>
                  (... all your commandline before run the job)
```

After that, you can run ```tensorboard --logdir=<checkpoint-path>``` to view Tensorboard of the job.

### Get component logs from a training job

We can use `yarn logs -applicationId <applicationId>` to get logs from CLI.
Or from YARN UI:

![alt text](../assets/job-logs-ui.png "Job logs UI")

<br />

## Submarine Commandline options

```$xslt
usage: ... job run

 -framework <arg>             Framework to use.
                              Valid values are: tensorflow, pytorch.
                              The default framework is Tensorflow.
 -checkpoint_path <arg>       Training output directory of the job, could
                              be local or other FS directory. This
                              typically includes checkpoint files and
                              exported model
 -docker_image <arg>          Docker image name/tag
 -env <arg>                   Common environment variable of worker/ps
 -input_path <arg>            Input of the job, could be local or other FS
                              directory
 -name <arg>                  Name of the job
 -num_ps <arg>                Number of PS tasks of the job, by default
                              it's 0
 -num_workers <arg>           Numnber of worker tasks of the job, by
                              default it's 1
 -ps_docker_image <arg>       Specify docker image for PS, when this is
                              not specified, PS uses --docker_image as
                              default.
 -ps_launch_cmd <arg>         Commandline of worker, arguments will be
                              directly used to launch the PS
 -ps_resources <arg>          Resource of each PS, for example
                              memory-mb=2048,vcores=2,yarn.io/gpu=2
 -queue <arg>                 Name of queue to run the job, by default it
                              uses default queue
 -saved_model_path <arg>      Model exported path (savedmodel) of the job,
                              which is needed when exported model is not
                              placed under ${checkpoint_path}could be
                              local or other FS directory. This will be
                              used to serve.
 -tensorboard <arg>           Should we run TensorBoard for this job? By
                              default it's true
 -verbose                     Print verbose log for troubleshooting
 -wait_job_finish             Specified when user want to wait the job
                              finish
 -worker_docker_image <arg>   Specify docker image for WORKER, when this
                              is not specified, WORKER uses --docker_image
                              as default.
 -worker_launch_cmd <arg>     Commandline of worker, arguments will be
                              directly used to launch the worker
 -worker_resources <arg>      Resource of each worker, for example
                              memory-mb=2048,vcores=2,yarn.io/gpu=2
 -localization <arg>          Specify localization to remote/local
                              file/directory available to all container(Docker).
                              Argument format is "RemoteUri:LocalFilePath[:rw]"
                              (ro permission is not supported yet).
                              The RemoteUri can be a file or directory in local
                              or HDFS or s3 or abfs or http .etc.
                              The LocalFilePath can be absolute or relative.
                              If relative, it'll be under container's implied
                              working directory.
                              This option can be set mutiple times.
                              Examples are
                              -localization "hdfs:///user/yarn/mydir2:/opt/data"
                              -localization "s3a:///a/b/myfile1:./"
                              -localization "https:///a/b/myfile2:./myfile"
                              -localization "/user/yarn/mydir3:/opt/mydir3"
                              -localization "./mydir1:."
 -conf <arg>                  User specified configuration, as
                              key=val pairs.
```

#### Notes:
When using `localization` option to make a collection of dependency Python
scripts available to entry python script in the container, you may also need to
set `PYTHONPATH` environment variable as below to avoid module import error
reported from `entry_script.py`.

```shell
... job run
  # the entry point
  --localization entry_script.py:<path>/entry_script.py
  # the dependency Python scripts of the entry point
  --localization other_scripts_dir:<path>/other_scripts_dir
  # the PYTHONPATH env to make dependency available to entry script
  --env PYTHONPATH="<path>/other_scripts_dir"
  --worker_launch_cmd "python <path>/entry_script.py ..."
```

<br />

## Build From Source

If you want to build the Submarine project by yourself, you can follow it [here](../development/BuildFromCode.md)
