---
title: Submarine Launcher
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

:::warning
Please note that this design doc is working-in-progress and need more works to complete. 
:::

## Introduction
Submarine is built and run in Cloud Native, taking advantage of the cloud computing model.

To give full play to the advantages of cloud computing. 
These applications are characterized by rapid and frequent build, release, and deployment. 
Combined with the features of cloud computing, they are decoupled from the underlying hardware and operating system, 
and can easily meet the requirements of scalability, availability, and portability. And provide better economy.

In the enterprise data center, submarine can support k8s/yarn/docker three resource scheduling systems; 
in the public cloud environment, submarine can support these cloud services in GCE/AWS/Azure;

## Requirement

### Cloud-Native Service

The submarine server is a long-running services in the daemon mode. 
The submarine server is mainly used by algorithm engineers to provide online front-end functions such as algorithm development, 
algorithm debugging, data processing, and workflow scheduling. 
And submarine server also mainly used for back-end functions such as scheduling and execution of jobs, tracking of job status, and so on.

Through the ability of rolling upgrades, we can better provide system stability. 
For example, we can upgrade or restart the workbench server without affecting the normal operation of submitted jobs.

You can also make full use of system resources.
For example, when the number of current developers or job tasks increases,
The number of submarine server instances can be adjusted dynamically.

In addition, submarine will provide each user with a completely independent workspace container. 
This workspace container has already deployed the development tools and library files commonly used by algorithm engineers including their operating environment. 
Algorithm engineers can work in our prepared workspaces without any extra work.

Each user's workspace can also be run through a cloud service.

### Service discovery
With the cluster function of submarine, each service only needs to run in the container, 
and it will automatically register the service in the submarine cluster center. 
Submarine cluster management will automatically maintain the relationship between service and service, service and user.

## Design

![cloud-service](../../assets/design/multi-dc-cloud.png)


### Launcher

The submarine launcher module defines the complete interface. 
By using this interface, you can run the submarine server, and workspace in k8s / yarn / docker / AWS / GCE / Azure.


### Launcher On Docker
In order to allow some small and medium-sized users without k8s/yarn to use submarine, 
we support running the submarine system in docker mode.

Users only need to provide several servers with docker runtime environment. 
The submarine system can automatically cluster these servers into clusters, manage all the hardware resources of the cluster, 
and run the service or workspace container in this cluster through scheduling algorithms.


### Launcher On Kubernetes

submarine operator

### Launcher On Yarn
[TODO]

### Launcher On AWS
[TODO]

### Launcher On GCP
[TODO]

### Launcher On Azure
[TODO]
