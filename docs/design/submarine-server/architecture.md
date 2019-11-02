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
# Submarine Server

## Motivation
Nowadays we use the client to submit the machine learning job and the cluster config is stored at the client. It’s insecure and difficult to use. So we should hide the underlying cluster system such as YARN/K8s.

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

## submarine-server

### REST
The REST API Service handles HTTP requests and is responsible for authentication. It acts as the caller for the JobManager component.

The REST component defines the generic job spec which describes the detailed info about job. For more details, refer to [here](https://docs.google.com/document/d/1kd-5UzsHft6gV7EuZiPXeWIKJtPqVwkNlqMvy0P_pAw/edit#).

### JobManager
The JobManager receives the job requests, persisting the job metadata in a database(MySQL in production), submitting and monitoring the job. Using the plug-in design pattern for submitter to extends more features. Submitting the job to cluster resource management system through the specified submitter plug-in. 

The JobManager has two main components: plug-ins manager and job monitor.

#### PlugMgr
The plug-ins manager is responsible for launching the submitter plug-ins, users have to add their jars to submarine-server’s classpath directly, thus put them on the system classloader. But if there are any conflicts between the dependencies introduced by the submitter plug-ins and the submarine-server itself, they can break the submarine-server, the plug-ins manager, or both. To solve this issue, we can instantiate submitter plug-ins using a classloader that is different from the system classloader.

#### Monitor
The monitor tracks the training life cycle and records the main events and key info in runtime. As the training job progresses, the metrics are needed for evaluation of the ongoing success or failure of the training progress. Due to adapt the different cluster resource management system, so we need a generic metric info structure and each submitter plug-in should inherit and complete it by itself.

### Submitter Plug-ins
Each plug-in uses a separate module under the server-submitter module. As the default implements, we provide for YARN and K8s. For YARN cluster, we provide the submitter-yarn and submitter-yarnservice plug-ins. The submitter-yarn plug-in used the [TonY](https://github.com/linkedin/TonY) as the runtime to run the training job, and the submitter-yarnservice plug-in direct use the [YARN Service](https://hadoop.apache.org/docs/stable/hadoop-yarn/hadoop-yarn-site/yarn-service/Overview.html) which supports  Hadoop v3.1 above. The submitter-k8s plug-in is used to submit the job to Kubernetes cluster and use the [operator](https://kubernetes.io/docs/concepts/extend-kubernetes/operator/) as the runtime. The submitter-k8s plug-in implements the operation of CRD object and provides the java interface. In the beginning, we use the [tf-operator](https://github.com/kubeflow/tf-operator) for the TensorFlow.

If Submarine want to support the other resource management system in the future, such as submarine-docker-cluster (submarine uses the Raft algorithm to create a docker cluster on the docker runtime environment on multiple servers, providing the most lightweight resource scheduling system for small-scale users). We should create a new plug-in module named submitter-docker under the server-submitter module.

### Failure Recovery
Use the database(MySQL in production) to do the submarine-server failure recovery. 

## submarine-client
The CLI client is the default implements for submarine-server RESTful API, it provides all the features which declared in the RESTful API.
