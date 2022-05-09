---
title: Implementation Notes
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

Before digging into details of implementations, you should read [architecture-and-requirements](./architecture-and-requirements.md) first to understand overall requirements and architecture.

Here're sub topics of Submarine implementations:

- [Submarine Storage](./storage-implementation.md): How to store metadata, logs, metrics, etc. of Submarine.
- [Submarine Environment](./environments-implementation.md): How environments created, managed, stored in Submarine.
- [Submarine Experiment](./experiment-implementation.md): How experiments managed, stored, and how the predefined experiment template works.
- [Submarine Notebook](./notebook-implementation.md): How experiments managed, stored, and how the predefined experiment template works.
- [Submarine Server](./submarine-server/architecture.md): How Submarine server is designed, architecture, implementation notes, etc.

Working-in-progress designs, Below are designs which are working-in-progress, we will move them to the upper section once design & review is finished:

- [Submarine HA Design](./wip-designs/submarine-clusterServer.md): How Submarine HA can be achieved, using RAFT, etc.
- [Submarine services deployment module:](./wip-designs/submarine-launcher.md) How to deploy submarine services to k8s or cloud.
