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
# Docs index

Click below contents if you want to understand more.

## Quick Start Guide

[Quick Start Guide](./helper/QuickStart.md)

## Build From Code

[Build From Code Guide](./development/BuildFromCode.md)

## Apache Hadoop Submarine Community

[Apache Hadoop Submarine Community Guide](./community/README.md)

## Submarine Workbench

[Submarine Workbench Guide](./workbench/README.md)

## Examples

Here're some examples about Submarine usage.

[Running Distributed CIFAR 10 Tensorflow Job](./helper/RunningDistributedCifar10TFJobs.md)

[Running Standalone CIFAR 10 PyTorch Job](./helper/RunningSingleNodeCifar10PTJobs.md)

## Development

[Submarine Project Development Guide](./development/README.md)

[Submarine Project Database Guide](./database/README.md)

## Dockerfile

[How to write Dockerfile for Submarine TensorFlow jobs](./helper/WriteDockerfileTF.md)

[How to write Dockerfile for Submarine PyTorch jobs](./helper/WriteDockerfilePT.md)

## Install Dependencies

**Note: You need to install dependencies when using hadoop yarn 3.1.x + or above.**

Submarine project may uses YARN Service (When Submarine YARN service runtime is being used, see [QuickStart](./helper/QuickStart.md), Docker container, and GPU (when GPU hardware available and properly configured).

That means as an admin, you may have to properly setup YARN Service related dependencies, including:

- YARN Registry DNS

Docker related dependencies, including:

- Docker binary with expected versions.
- Docker network which allows Docker container can talk to each other across different nodes.

And when GPU plan to be used:

- GPU Driver.
- Nvidia-docker.

For your convenience, we provided installation documents to help you to setup your environment. You can always choose to have them installed in your own way.

Use Submarine installer to install dependencies: [EN](../submarine-installer/README.md) [CN](../submarine-installer/README-CN.md)

Alternatively, you can follow manual install dependencies: [EN](./helper/InstallationGuide.md) [CN](./helper/InstallationGuideChineseVersion.md)

Once you have installed dependencies, please follow following guide to [TestAndTroubleshooting](./helper/TestAndTroubleshooting.md).

