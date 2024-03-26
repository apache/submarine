---
title: Environments Implementation
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

## Overview

Environment profiles (or environment for short) defines a set of libraries and when Docker is being used, a Docker image in order to run an experiment or a notebook. 

Docker and/or VM-image (such as, VirtualBox/VMWare images, Amazon Machine Images - AMI, Or custom image of Azure VM) defines the base layer of the environment. Please note that VM-image is different from VM instance type,

On top of that, users can define a set of libraries (such as Python/R) to install, we call it kernel.

**Example of Environment**

```

     +-------------------+
     |+-----------------+|
     || Python=3.7      ||
     || Tensorflow=2.0  ||
     |+---Exp Dependency+|
     |+-----------------+|
     ||OS=Ubuntu16.04   ||
     ||CUDA=10.2        ||
     ||GPU_Driver=375.. ||
     |+---Base Library--+|
     +-------------------+
```

As you can see, There're base libraries, such as what OS, CUDA version, GPU driver, etc. They can be achieved by specifying a VM-image / Docker image.

On top of that, user can bring their dependencies, such as different version of Python, Tensorflow, Pandas, etc.

**How users use environment?**

Users can save different environment configs which can be also shared across the platform. Environment profiles can be used to run a notebook (e.g. by choosing different kernel from Jupyter), or an experiment. Predefined experiment library includes what environment to use so users don't have to choose which environment to use.

```

        +-------------------+
        |+-----------------+|       +------------+
        || Python=3.7      ||       |User1       |
        || Tensorflow=2.0  ||       +------------+
        |+---Kernel -------+|       +------------+
        |+-----------------+|<----+ |User2       |
        ||OS=Ubuntu16.04   ||     + +------------+
        ||CUDA=10.2        ||     | +------------+
        ||GPU_Driver=375.. ||     | |User3       |
        |+---Base Library--+|     | +------------+
        +-----Default-Env---+     |
                                  |
                                  |
        +-------------------+     |
        |+-----------------+|     |
        || Python=3.3      ||     |
        || Tensorflow=2.0  ||     |
        |+---kernel--------+|     |
        |+-----------------+|     |
        ||OS=Ubuntu16.04   ||     |
        ||CUDA=10.3        ||<----+
        ||GPU_Driver=375.. ||
        |+---Base Library--+|
        +-----My-Customized-+
```

There're two environments in the above graph, "Default-Env" and "My-Customized", which can have different combinations of libraries for different experiments/notebooks. Users can choose different environments for different experiments as they want.

Environments can be added/listed/deleted/selected through CLI/SDK/UI.

# Implementation

## Environment API definition

Let look at what object definition looks like to define an environment, API of environment looks like:

```
    name: "my_submarine_env",
    vm-image: "...",
    docker-image: "...", 
    kernel: 
       <object of kernel>
    description: "this is the most common env used by team ABC"
```

- `vm-image` is optional if we don't need to launch new VM (like running a training job in a cloud-remote machine). 
- `docker-image` is required
- `kernel` could be optional if kernel is already included by vm-image or docker-image.
- `name` of the environment should be unique in the system, so user can reference it when create a new experiment/notebook.

## VM-image and Docker-image

Docker-image and VM image should be prepared by system admin / SREs, it is hard for Data-Scientists to write an error-proof Dockerfile, and push/manage Docker images. This is one of the reason we hide Docker-image inside "environment", we will encourage users to customize their kernels if needed, but don't have to touch Dockerfile and build/push/manage new Docker images.

As a project, we will document what's the best practice and example of Dockerfiles. 

Dockerfile should include proper `ENTRYPOINT` definition which pointed to our default script, so no matter it is notebook, or an experiment, we will setup kernel (see below) and other environment variables properly.

## Kernel Implementation

After investigating different alternatives (such as pipenv, venv, etc.), we decided to use Conda environment which nicely replaces Python virtual env, pip, and can also support other languages. More details can be found at: https://medium.com/@krishnaregmi/pipenv-vs-virtualenv-vs-conda-environment-3dde3f6869ed

When once Conda, users can easily add, remove dependency of a Conda environment. User can also easily export environment to yaml file.

The yaml file of Conda environment by using `conda env export` looks like: 

```
name: base
channels:
  - defaults
dependencies:
  - _ipyw_jlab_nb_ext_conf=0.1.0=py37_0
  - alabaster=0.7.12=py37_0
  - anaconda=2020.02=py37_0
  - anaconda-client=1.7.2=py37_0
  - anaconda-navigator=1.9.12=py37_0
  - anaconda-project=0.8.4=py_0
  - applaunchservices=0.2.1=py_0
```

Including Conda kernel, the environment object may look like: 

```
name: "my_submarine_env",
    vm-image: "...",
    docker-image: "...", 
    kernel: 
      name: team_default_python_3.7
      channels:
        - defaults
      dependencies:
        - _ipyw_jlab_nb_ext_conf=0.1.0=py37_0
        - alabaster=0.7.12=py37_0
        - anaconda=2020.02=py37_0
        - anaconda-client=1.7.2=py37_0
        - anaconda-navigator=1.9.12=py37_0
```

When launch a new experiment / notebook session using the `my_submarine_env`, submarine server will use defined Docker image, and Conda kernel to launch of container. 

## Storage of Environment 

Environment of Submarine is just a simple text file, so it will be persisted in Submarine metastore, which is ideally a Database. 

Docker image is stored inside a regular Docker registry, which will be handled outside of the system. 

Conda dependencies are stored in Conda channel (where referenced packages are stored), which will be handled/setuped separately. (Popular conda channels are `default` and `conda-forge`)

For more detailed discussion about storage-related implementations, please refer to [storage-implementation](./storage-implementation).

## How to implement to make user can easily use Submarine environments? 

We like simplicities, and we don't want to leak complexities of implementations to the users. To make it happen, we have to do some works to hide complexities. 

There're two primary uses of environments: experiments and notebook, for both of them, users should not do works like explictily call `conda active $env_name` to active environments. To make it happen, what we can do is to include following parts in Dockerfile 

```
FROM ubuntu:18.04

<Include whatever base-libraries like CUDA, etc.>

<Make sure conda (with our preferred version) is installed>
<Make sure Jupyter (with our preferred version) is installed>

# This is just a sample of Dockerfile, users can do more customizations if needed
ENTRYPOINT ["/submarine-bootstrap.sh"]
```

When Submarine Server (this is implementation detail of Submarine Server, user will not see it at all) launch an experiment, or notebook, it will invoke following `docker run` command (or any other equvilant like using K8s spec): 

```
docker run <submarine_docker_image> --kernel <kernel_name> -- .... python train.py --batch_size 5 (and other parameters)
```

Similarily, to launch a notebook: 

```
docker run <submarine_docker_image> --kernel <kernel_name> -- .... jupyter
```

The `submarine-bootstrap.sh` is part of Submarine repo, and will handle `--kernel` argument which will invoke  `conda active $kernel_name` before anything else. (Like run the training job).



