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
# User Document of Submarine

This is index of user document of Submarine.

## Build From Code

[Build From Code Guide](./development/BuildFromCode.md)

FIXME: Where's build for K8s?

## Quick Start Guide

[Quick Start Guide](./helper/QuickStart.md)

## Submarine Server
[Submarine Server Guide](./submarine-server/README.md)

## Examples

Here're some examples about Submarine usage.

[Running Distributed CIFAR 10 Tensorflow Job_With_Yarn_Service_Runtime](helper/RunningDistributedCifar10TFJobsWithYarnService.md)

[Running Standalone CIFAR 10 PyTorch Job_With_Yarn_Service_Runtime](helper/RunningSingleNodeCifar10PTJobsWithYarnService.md)

[Running Distributed thchs30 Kaldi Job](./ecosystem/kaldi/RunningDistributedThchs30KaldiJobs.md)

## Dockerfile

[How to write Dockerfile for Submarine TensorFlow jobs](./helper/WriteDockerfileTF.md)

[How to write Dockerfile for Submarine PyTorch jobs](./helper/WriteDockerfilePT.md)

[How to write Dockerfile for Submarine MXNet jobs](./helper/WriteDockerfileMX.md)

[How to write Dockerfile for Submarine Kaldi jobs](./ecosystem/kaldi/WriteDockerfileKaldi.md)

## Install Dependencies

**Note: You need to install dependencies when using Hadoop YARN 3.1.x + or above.**

Submarine project may use YARN Service (When Submarine YARN service runtime is being used, see [QuickStart](./helper/QuickStart.md), Docker container, and GPU (when GPU hardware available and properly configured).

That means as an admin you may have to properly setup YARN Service related dependencies, including:

- YARN Registry DNS

Docker related dependencies, including:

- Docker binary with expected versions.
- Docker network which allows Docker container can talk to each other across different nodes.

And when GPU plans to be used:

- GPU Driver.
- Nvidia-docker.

For your convenience, we provide installation documents to help you to set up your environment. You can always choose to have them installed in your own way.

Use Submarine installer to install dependencies: [EN](../dev-support/submarine-installer/README.md) [CN](../dev-support/submarine-installer/README-CN.md)

Alternatively, you can follow manual install dependencies: [EN](./helper/InstallationGuide.md) [CN](./helper/InstallationGuideChineseVersion.md)

Once you have installed dependencies, please follow following guide to [TestAndTroubleshooting](./helper/TestAndTroubleshooting.md).

