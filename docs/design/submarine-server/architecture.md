<!--
  Licensed to the Apache Software Foundation (ASF) under one
  or more contributor license agreements.  See the NOTICE file
  distributed with this work for additional information
  regarding copyright ownership.  The ASF licenses this file
  to you under the Apache License, Version 2.0 (the
  "License"); you may not use this file except in compliance
  with the License.  You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing,
  software distributed under the License is distributed on an
  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
  KIND, either express or implied.  See the License for the
  specific language governing permissions and limitations
  under the License.
  -->
# Submarine Server Architecture And Implementation

## Architecture Overview

```
    +---------------Submarine Server ---+
    |                                   |
    | +------------+ +------------+     |
    | |Web Svc/Prxy| |Backend Svc |     |    +--Submarine Asset +
    | +------------+ +------------+     |    |Project/Notebook  |
    |   ^         ^                     |    |Model/Metrics     |
    +---|---------|---------------------+    |Libraries/Dataset |
        |         |                          +------------------+
        |         |
        |      +--|-Compute Cluster 1---+    +--Image Registry--+
        +      |  |                     |    |   User's Images  |
      User /   |  +                     |    |                  |
      Admin    | User Notebook Instance |    +------------------+
               | Experiment Runs        |
               +------------------------+    +-Data Storage-----+
                                             | S3/HDFS, etc.    |
               +----Compute Cluster 2---+    |                  |
                                             +------------------+
                        ...
```

Here's a diagram to illustrate the Submarine's deployment.

- Submarine Server consists of web service/proxy, and backend services. They're like "control planes" of Submarine, and users will interact with these services.
- Submarine server could be a microservice architecture and can be deployed to one of the compute clusters. (see below, this will be useful when we only have one cluster). 
- There're multiple compute clusters that could be used by Submarine service. For user's running notebook instance, jobs, etc. they will be placed to one of the compute clusters by user's preference or defined policies.
- Submarine's asset includes project/notebook(content)/models/metrics/dataset-meta, etc. can be stored inside Submarine's own database.
- Datasets can be stored in various locations such as S3/HDFS. 
- Users can push container (such as Docker) images to a preconfigured registry in Submarine, so Submarine service can know how to pull required container images.
- Image Registry/Data-Storage, etc. are outside of Submarine server's scope and should be managed by 3rd party applications.

## Submarine Server and its APIs

Submarine server is designed to allow data scientists to access notebooks, submit/manage jobs, manage models, create model training workflows, access datasets, etc.

Submarine Server exposed UI and REST API. Users can also use CLI / SDK to manage assets inside Submarine Server.

```
           +----------+
           | CLI      |+---+
           +----------+    v              +----------------+
                         +--------------+ | Submarine      |
           +----------+  | REST API     | |                |
           | SDK      |+>|              |+>  Server        |
           +----------+  +--------------+ |                |
                           ^              +----------------+
           +----------+    |
           | UI       |+---+
           +----------+
```

REST API will be used by the other 3 approaches. (CLI/SDK/UI) 

The REST API Service handles HTTP requests and is responsible for authentication. It acts as the caller for the JobManager component.

The REST component defines the generic job spec which describes the detailed info about job. For more details, refer to [here](https://docs.google.com/document/d/1kd-5UzsHft6gV7EuZiPXeWIKJtPqVwkNlqMvy0P_pAw/edit#). (Please note that we're converting REST endpoint description from Java-based REST API to swagger definition, once that is done, we should replace the link with swagger definition spec).

## Proposal
 ```
                                                                +---------------------+
 +-----------+                                                  | +--------+   +----+ |
 |           |                                                  | |runtime1+-->+job1| |
 | workbench +---+   +----------------------------------+       | +--------+   +----+ |
 |           |   |   | +------+ +---------------------+ |   +-->+ +--------+   +----+ |
 +-----------+   |   | |      | | +------+  +-------+ | |   |   | |runtime2+-->+job2| |
                 |   | |      | | | YARN |  |  K8s  | | |   |   | +--------+   +----+ |
 +-----------+   |   | |      | | +------+  +-------+ | |   |   |     YARN Cluster    |
 |           |   |   | |      | |      submitter      | |   |   +---------------------+
 |    CLI    +------>+ | REST | +---------------------+ +---+
 |           |   |   | |      | +---------------------+ |   |   +---------------------+
 +-----------+   |   | |      | | +-------+ +-------+ | |   |   | +--------+   +----+ |
                 |   | |      | | |PlugMgr| |monitor| | |   |   | |        +-->+job1| |
 +-----------+   |   | |      | | +-------+ +-------+ | |   |   | |        |   +----+ |
 |           |   |   | |      | |      JobManager     | |   +-->+ |operator|   +----+ |
 |    SDK    +---+   | +------+ +---------------------+ |       | |        +-->+job2| |
 |           |       +----------------------------------+       | +--------+   +----+ |
 +-----------+                                                  |     K8s Cluster     |
    client                          server                      +---------------------+
 ```
We propose to split the original core module in the old layout into two modules, CLI and server as shown in FIG. The submarine-client calls the REST APIs to submit and retrieve the job info. The submarine-server provides the REST service, job management, submitting the job to cluster, and running job in different clusters through the corresponding runtime.

## Submarine Server Components

```

   +----------------------Submarine Server--------------------------------+
   | +-----------------+ +------------------+ +--------------------+      |
   | |  Experiment     | |Notebook Session  | |Environment Mgr     |      |
   | |  Mgr            | |Mgr               | |                    |      |
   | +-----------------+ +------------------+ +--------------------+      |
   |                                                                      |
   | +-----------------+ +------------------+ +--------------------+      |
   | |  Model Registry | |Model Serving Mgr | |Compute Cluster Mgr |      |
   | |                 | |                  | |                    |      |
   | +-----------------+ +------------------+ +--------------------+      |
   |                                                                      |
   | +-----------------+ +------------------+ +--------------------+      |
   | | DataSet Mgr     | |User/Team         | |Metadata Mgr        |      |
   | |                 | |Permission Mgr    | |                    |      |
   | +-----------------+ +------------------+ +--------------------+      |
   +----------------------------------------------------------------------+
```

### Experiment Manager 

TODO

### Notebook Sessions Manager 

TODO

### Environment Manager

TODO

### Model Registry

TODO

### Model Serving Manager 

TODO

### Compute Cluster Manager

TODO

### Dataset Manager 

TODO

### User/team permissions manager 

TODO

### Metadata Manager

TODO

## Components/services outside of Submarine Server's scope

TODO: Describe what are the out-of-scope components, which should be handled and managed outside of Submarine server. Candidates are: Identity management, data storage, metastore storage, etc.

## Experiment-related components

#### Experiment Manager

The JobManager receives the job requests, persisting the job metadata in a database(MySQL in production), submitting and monitoring the job. Using the plug-in design pattern for submitter to extends more features. Submitting the job to cluster resource management system through the specified submitter plug-in. 

The JobManager has two main components: plug-ins manager and job monitor.

#### PlugMgr
The plug-ins manager is responsible for launching the submitter plug-ins, users have to add their jars to submarine-server’s classpath directly, thus put them on the system classloader. But if there are any conflicts between the dependencies introduced by the submitter plug-ins and the submarine-server itself, they can break the submarine-server, the plug-ins manager, or both. To solve this issue, we can instantiate submitter plug-ins using a classloader that is different from the system classloader.

#### Monitor
The monitor tracks the training life cycle and records the main events and key info in runtime. As the training job progresses, the metrics are needed for evaluation of the ongoing success or failure of the training progress. Due to adapt the different cluster resource management system, so we need a generic metric info structure and each submitter plug-in should inherit and complete it by itself.

#### Submitter Plug-ins
Each plug-in uses a separate module under the server-submitter module. As the default implements, we provide for YARN and K8s. For YARN cluster, we provide the submitter-yarn and submitter-yarnservice plug-ins. The submitter-yarn plug-in used the [TonY](https://github.com/linkedin/TonY) as the runtime to run the training job, and the submitter-yarnservice plug-in direct use the [YARN Service](https://hadoop.apache.org/docs/stable/hadoop-yarn/hadoop-yarn-site/yarn-service/Overview.html) which supports  Hadoop v3.1 above. The submitter-k8s plug-in is used to submit the job to Kubernetes cluster and use the [operator](https://kubernetes.io/docs/concepts/extend-kubernetes/operator/) as the runtime. The submitter-k8s plug-in implements the operation of CRD object and provides the java interface. In the beginning, we use the [tf-operator](https://github.com/kubeflow/tf-operator) for the TensorFlow.

If Submarine want to support the other resource management system in the future, such as submarine-docker-cluster (submarine uses the Raft algorithm to create a docker cluster on the docker runtime environment on multiple servers, providing the most lightweight resource scheduling system for small-scale users). We should create a new plug-in module named submitter-docker under the server-submitter module.

## Common modules of experiment/notebook-session/model-serving

#### 

#### Failure Recovery
Use the database(MySQL in production) to do the submarine-server failure recovery. 
