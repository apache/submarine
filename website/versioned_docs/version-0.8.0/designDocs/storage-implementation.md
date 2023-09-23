---
title: Storage Implementation
---
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

## ML-related objects and their storages

First let's look at what user will interact for most of the time: 

- Notebook 
- Experiment
- Model Servings

```


                              +---------+    +------------+
                              |Logs     |<--+|Notebook    |
      +----------+            +---------+    +------------+     +----------------+
      |Trackings |                        <-+|Experiment  |<--+>|Model Artifacts |
      +----------+     +-----------------+   +------------+     +----------------+
      +----------+<---+|ML-related Metric|<--+Servings    |
      |tf.events |     +-----------------+   +------------+
      +----------+                                 ^              +-----------------+
                                                   +              | Environments    |
                                        +----------------------+  |                 |
            +-----------------+         | Submarine Metastore  |  |  Dependencies   |
            |Code             |         +----------------------+  |                 |
            +-----------------+         |Experiment Meta       |  |   Docker Images |
                                        +----------------------+  +-----------------+
                                        |Model Store Meta      |
                                        +----------------------+
                                        |Model Serving Meta    |
                                        +----------------------+
                                        |Notebook meta         |
                                        +----------------------+
                                        |Experiment Templates  |
                                        +----------------------+
                                        |Environments Meta     |
                                        +----------------------+
```

First of all, all the notebook-sessions / experiments / model-serving instances) are more or less interact with following storage objects:

- Logs for these tasks for troubleshooting. 
- ML-related metrics such as loss, epoch, etc. (in contrast of system metrics such as CPU/memory usage, etc.)
  - There're different types of ML-related metrics, for Tensorflow/pytorch, they can use tf.events and get visualizations on tensorboard. 
  - Or they can use tracking APIs (such as Submarine tracking, mlflow tracking, etc.) to output customized tracking results for non TF/Pytorch workloads. 
- Training jobs of experiment typically generate model artifacts (files) which need persisted, and both of notebook, model serving needs to load model artifacts from persistent storage. 
- There're various of meta information, such as experiment meta, model registry, model serving, notebook, experiment, environment, etc. We need be able to read these meta information back.
- We also have code for experiment (like training/batch-prediction), notebook (ipynb), and model servings.
- And notebook/experiments/model-serving need depend on environments (dependencies such as pip, and Docker Images).

### Implementation considerations for ML-related objects

| Object Type                              | Characteristics                                              | Where to store                                               |
| ---------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Metrics: tf.events                       | Time series data with k/v, appendable to file                | Local/EBS, HDFS, Cloud Blob Storage                          |
| Metrics: other tracking metrics          | Time series data with k/v, appendable to file                | Local, HDFS, Cloud Blob Storage, Database                    |
| Logs                                     | Large volumes, #files are potentially huge.                  | Local (temporary), HDFS (need aggregation), Cloud Blob Storage |
| Submarine Metastore                      | CRUD operations for small meta data.                         | Database                                                     |
| Model Artifacts                          | Size varies for model (from KBs to GBs). #files are potentially huge. | HDFS, Cloud Blob Storage                                     |
| Code                                     | Need version control. (Please find detailed discussions below for code storage and localization) | Tarball on HDFS/Cloud Blog Storage, or Git                   |
| Environment (Dependencies, Docker Image) |                                                              | Public/private environment repo (like Conda channel), Docker registry. |

### Detailed discussions

#### Store code for experiment/notebook/model-serving

There're following ways to get experiment code: 

**1) Code is part of Git repo:** (***<u>Recommended</u>***)

This is our recommended approach, once code is part of Git, it will be stored in version control, any change will be tracked, and much easier for users to trace back what change triggered a new bug, etc.

**2) Code is part of Docker image:** 

***This is an anti-pattern and we will NOT recommend you to use it***, Docker image can be used to include ANYTHING, like dependencies, the code you will execute, or even data. But this doesn't mean you should do it. We recommend to use Docker image ONLY for libraries/dependencies.

Making code to be part of Docker image makes hard to edit code (if you want to update a value in your Python file, you will have to recreate the Docker image, push it and rerun it).

**3) Code is part of S3/HDFS/ABFS:** 

User may want to store their training code to a tarball on a shared storage. Submarine need to download code from remote storage to the launched container before running the code. 

#### Localization of experiment/notebook/model-serving code

To make user experiences keeps same across different environment, we will localize code to a same folder after the container is launched, preferably `/code`

For example, there's a git repo need to be synced up for an experiment/notebook/model-serving (example above):

```
experiment: #Or notebook, model-serving
       name: "abc",
       environment: "team-default-ml-env"
       ... (other fields)
			 code:
   	       sync_mode: git
           url: "https://foo.com/training-job.git" 
```

After localize, `training-job/` will be placed under `/code` 

When we running on K8s environment, we can use K8s's initContainer and emptyDir to do these things for us. K8s POD spec (generated by Submarine server instead of user, user should NEVER edit K8s spec, that's too unfriendly to data-scientists): 

```
apiVersion: v1
kind: Pod
metadata:
  name: experiment-abc
spec:
  containers:
  - name: experiment-task
    image: training-job
    volumeMounts:
    - name: code-dir
      mountPath: /code
  initContainers:
  - name: git-localize
    image: git-sync
    command: "git clone .. /code/"
    volumeMounts:
    - name: code-dir
      mountPath: /code
  volumes:
  - name: code-dir
    emptyDir: {}
```

The above K8s spec create a code-dir and mount it to `/code` to launched containers. The initContainer `git-localize` uses `https://github.com/kubernetes/git-sync` to do the sync up. (If other storages are used such as s3, we can use similar initContainer approach to download contents)

## System-related metrics/logs and their storages

Other than ML-related objects, we have system-related objects, including: 

- Daemon logs (like logs of Submarine server). 
- Logs for other dependency components (like Kubernetes logs when running on K8s). 
- System metrics (Physical resource usages by daemons, launched training containers, etc.). 

All these information should be handled by 3rd party system, such as Grafana, Prometheus, etc. And system admins are responsible to setup these infrastructures, dashboard. Users of submarine should NOT interact with system related metrics/logs. It is system admin's responsibility.

## Attachable Volumes 

It is possible user has needs to have an attachable volume for their experiment / notebook, this is especially useful for notebook storage, since contents of notebook can be automatically saved, and it can be used as user's home folder. 

Downside of attachable volume is, it is not versioned, even notebook is mainly used for adhoc exploring tasks, an unversioned notebook file can lead to maintenance issues in the future. 

Since this is a common requirement, we can consider to support attachable volumes in Submarine in a long run, but with relatively lower priority.

## In-scope / Out-of-scope 

 Describe what Submarine project should own and what Submarine project should NOT own.

