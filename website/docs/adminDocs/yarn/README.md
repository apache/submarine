---
title: Running Submarine on YARN
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

Submarine for YARN supports TensorFlow, PyTorch and MXNet framework. (Which is leveraging [TonY](https://github.com/linkedin/TonY) created by Linkedin to run deep learning training jobs on YARN.

Submarine also supports GPU-on-YARN and Docker-on-YARN feature.

Submarine can run on Hadoop 2.7.3 or later version, if GPU-on-YARN or Docker-on-YARN feature is needed, newer Hadoop version is required, please refer to the next section about what Hadoop version to choose.

## Hadoop version

Must:

- Apache Hadoop version newer than 2.7.3

Optional:

- When you want to use GPU-on-YARN feature with Submarine, please make sure Hadoop is at least 2.10.0+ (or 3.1.0+), and follow [Enable GPU on YARN 2.10.0+](https://hadoop.apache.org/docs/r2.10.0/hadoop-yarn/hadoop-yarn-site/UsingGpus.html) to enable GPU-on-YARN feature.
- When you want to run training jobs with Docker container, please make sure Hadoop is at least 2.8.2, and follow [Enable Docker on YARN 2.8.2+](https://hadoop.apache.org/docs/r2.8.2/hadoop-yarn/hadoop-yarn-site/DockerContainers.html) to enable Docker-on-YARN feature.

## Submarine YARN Runtime Guide

[YARN Runtime Guide](../../userDocs/yarn/YARNRuntimeGuide) talk about how to use Submarine to run jobs on YARN, with Docker / without Docker.
