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
- [Enable GPU on YARN 2.10.0+](https://hadoop.apache.org/docs/r2.10.0/hadoop-yarn/hadoop-yarn-site/UsingGpus.html)
- [Enable Docker on YARN 2.8.2+](https://hadoop.apache.org/docs/r2.8.2/hadoop-yarn/hadoop-yarn-site/DockerContainers.html)
- [Build Docker images](WriteDockerfileTF.md)


<br />

## Submarine Configuration

After submarine 0.2.0, it supports two runtimes which are YARN service runtime and Linkedin's TonY runtime for YARN. Each runtime can support TensorFlow, PyTorch and MXNet framework. But we don't need to worry about the usage because the two runtime implements the same interface.

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


For a quick try on Cifar10 example with YARN native service runtime, check [YARN Service Cifar10 Example](RunningDistributedCifar10TFJobsWithYarnService.md)


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
usage: job run
 -checkpoint_path <arg>            Training output directory of the job,
                                   could be local or other FS directory.
                                   This typically includes checkpoint
                                   files and exported model
 -conf <arg>                       User specified configuration, as
                                   key=val pairs.
 -distribute_keytab                Distribute local keytab to cluster
                                   machines for service authentication. If
                                   not specified, pre-distributed keytab
                                   of which path specified by
                                   parameterkeytab on cluster machines
                                   will be used
 -docker_image <arg>               Docker image name/tag
 -env <arg>                        Common environment variable of
                                   worker/ps
 -f <arg>                          Config file (in YAML format)
 -framework <arg>                  Framework to use. Valid values are:
                                   tensorlow,pytorch,mxnet! The default
                                   framework is tensorflow.
 -h,--help                         Print help
 -input_path <arg>                 Input of the job, could be local or
                                   other FS directory
 -insecure                         Cluster is not Kerberos enabled.
 -keytab <arg>                     Specify keytab used by the job under
                                   security environment
 -localization <arg>               Specify localization to make
                                   remote/local file/directory available
                                   to all container(Docker). Argument
                                   format is "RemoteUri:LocalFilePath[:rw]
                                   " (ro permission is not supported yet)
                                   The RemoteUri can be a file or
                                   directory in local or HDFS or s3 or
                                   abfs or http .etc. The LocalFilePath
                                   can be absolute or relative. If it's a
                                   relative path, it'll be under
                                   container's implied working directory
                                   but sub directory is not supported yet.
                                   This option can be set mutiple times.
                                   Examples are
                                   -localization
                                   "hdfs:///user/yarn/mydir2:/opt/data"
                                   -localization "s3a:///a/b/myfile1:./"
                                   -localization
                                   "https:///a/b/myfile2:./myfile"
                                   -localization
                                   "/user/yarn/mydir3:/opt/mydir3"
                                   -localization "./mydir1:."
 -name <arg>                       Name of the job
 -num_ps <arg>                     Number of PS tasks of the job, by
                                   default it's 0. Can be used with
                                   TensorFlow or MXNet frameworks.
 -num_schedulers <arg>             Number of scheduler tasks of the job.
                                   It should be 1 or 0, by default it's
                                   0.Can only be used with MXNet
                                   framework.
 -num_workers <arg>                Number of worker tasks of the job, by
                                   default it's 1.Can be used with
                                   TensorFlow or PyTorch or MXNet
                                   frameworks.
 -principal <arg>                  Specify principal used by the job under
                                   security environment
 -ps_docker_image <arg>            Specify docker image for PS, when this
                                   is not specified, PS uses
                                   --docker_image as default.Can be used
                                   with TensorFlow or MXNet frameworks.
 -ps_launch_cmd <arg>              Commandline of PS, arguments will be
                                   directly used to launch the PSCan be
                                   used with TensorFlow or MXNet
                                   frameworks.
 -ps_resources <arg>               Resource of each PS, for example
                                   memory-mb=2048,vcores=2,yarn.io/gpu=2Ca
                                   n be used with TensorFlow or MXNet
                                   frameworks.
 -queue <arg>                      Name of queue to run the job, by
                                   default it uses default queue
 -quicklink <arg>                  Specify quicklink so YARNweb UI shows
                                   link to given role instance and port.
                                   When --tensorboard is specified,
                                   quicklink to tensorboard instance will
                                   be added automatically. The format of
                                   quick link is: Quick_link_label=http(or
                                   https)://role-name:port. For example,
                                   if want to link to first worker's 7070
                                   port, and text of quicklink is
                                   Notebook_UI, user need to specify
                                   --quicklink
                                   Notebook_UI=https://master-0:7070
 -saved_model_path <arg>           Model exported path (savedmodel) of the
                                   job, which is needed when exported
                                   model is not placed under
                                   ${checkpoint_path}could be local or
                                   other FS directory. This will be used
                                   to serve.
 -scheduler_docker_image <arg>     Specify docker image for scheduler,
                                   when this is not specified, scheduler
                                   uses --docker_image as default. Can
                                   only be used with MXNet framework.
 -scheduler_launch_cmd <arg>       Commandline of scheduler, arguments
                                   will be directly used to launch the
                                   scheduler. Can only be used with MXNet
                                   framework.
 -scheduler_resources <arg>        Resource of each scheduler, for example
                                   memory-mb=2048,vcores=2. Can only be
                                   used with MXNet framework.
 -tensorboard                      Should we run TensorBoard for this job?
                                   By default it's disabled.Can only be
                                   used with TensorFlow framework.
 -tensorboard_docker_image <arg>   Specify Tensorboard docker image. when
                                   this is not specified, Tensorboard uses
                                   --docker_image as default.Can only be
                                   used with TensorFlow framework.
 -tensorboard_resources <arg>      Specify resources of Tensorboard, by
                                   default it is memory=4G,vcores=1.Can
                                   only be used with TensorFlow framework.
 -verbose                          Print verbose log for troubleshooting
 -wait_job_finish                  Specified when user want to wait the
                                   job finish
 -worker_docker_image <arg>        Specify docker image for WORKER, when
                                   this is not specified, WORKER uses
                                   --docker_image as default.Can be used
                                   with TensorFlow or PyTorch or MXNet
                                   frameworks.
 -worker_launch_cmd <arg>          Commandline of worker, arguments will
                                   be directly used to launch the
                                   workerCan be used with TensorFlow or
                                   PyTorch or MXNet frameworks.
 -worker_resources <arg>           Resource of each worker, for example
                                   memory-mb=2048,vcores=2,yarn.io/gpu=2Ca
                                   n be used with TensorFlow or PyTorch or
                                   MXNet frameworks.
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
