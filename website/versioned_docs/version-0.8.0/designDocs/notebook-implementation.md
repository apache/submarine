---
title: Notebook Implementation
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

### User's interaction

Users can start N (N >= 0) number of Notebook sessions, a notebook session is a running notebook instance.

- Notebook session can be launched by Submarine UI (P0), and Submarine CLI (P2). 
- When launch notebook session, users can choose T-shirt size of notebook session (how much mem/cpu/gpu resources, or resource profile such as small, medium, large, etc.). (P0)
- And user can choose an environment for notebook. More details please refer to [environment implementation](./environments-implementation.md) (P0)
- When start a notebook, user can choose what code to be initialized, similar to experiment. (P1)
- Optionally, users can choose to attach a persistent volume to a notebook session. (P2)

Users can get a list of notebook sessions belongs to themselves, and connect to notebook session. 

User can choose to terminate a running notebook session.

### Admin's interaction 

- How many concurrent notebook sessions can be launched by each user is determined by resource quota limits of each user, and maximum concurrent notebook sessions can be launched by each user. (P2)

## Relationship with other components

### Metadata store

Running notebook sessions' metadata need persistented in Submarine's metadata store (Database).

### Submarine Server

```

  +--------------+  +--------Submarine Server--------------------+
  |Submarine UI  |  | +-------------------+                      |
  |              |+--->  Submarine        |                      |
  |  Notebook    |  | |  Notebook REST API|                      |
  +--------------+  | |                   |                      |
                    | +--------+----------+     +--------------+ |
                    |          |             +->|Metastore     | |
                    | +--------v----------+  |  |DB            | |
                    | | Submarine         +--+  +--------------+ |
                    | | Notebook Mgr      |                      |
                    | |                   |                      |
                    | |                   |                      |
                    | +--------+----------+                      |
                    |          |                                 |
                    +----------|---------------------------------+
                               |
                +--------------+
       +--------v---------+
       | Notebook Session |
       |                  |
       |   instance       |
       |                  |
       +------------------+
```

Once user use Submarine UI to launch a notebook session, Submarine notebook manager inside Submarine Server will persistent notebook session's metadata, and launch a new notebook session instance. 

### Resource manager

When using K8s as resource manager, Submarine notebook session will run as a new POD.

### Storage

There're several different types of storage requirements for Submarine notebook. 

For code, environment, etc, storage, please refer to [storage implementation](./storage-implementation.md), check "Localization of experiment/notebook/model-serving code".

When there're needs to attach volume (such as user's home folder) to Submarine notebook session, please check [storage implementation](./storage-implementation.md), check "Attachable volume".

### Environment

Submarine notebook's environment should be used to run experiment, model serving, etc. Please check [environment implementation](./environments-implementation.md). (More specific to notebook, please check "How to implement to make user can easily use Submarine environments")

Please note that notebook's Environment should include right version of notebook libraries, and admin should follow the guidance to build correct Docker image, Conda libraries to correctly run Notebook.

### Submarine SDK (For Experiment, etc.)

Users can run new experiment, access metrics information, or do model operations using Submarine SDK. 

Submarine SDK is a Python library which can talk to Submarine Server which need Submarine Server's endpoint as well as user credentials.

To ensure better experience, we recommend always install proper version of Submarine SDK from environment which users can use Submarine SDK directly from commandline. (We as Submarine community can provide sample Dockerfile or Conda environment which have correct base libraries installed for Submarine SDK).

Submarine Server IP will be configured automatically by Submarine Server, and added as an envar when Submarine notebook session got launched.

### Security 

Please refer to [Security Implementation](./wip-designs/security-implementation.md)

Once user accessed to a running notebook session, the user can also access resources of the notebook, capability of submit new experiment, and access data. This is also very dangerous so we have to protect it. 

A simple solution is to use token-based authentication https://jupyter-notebook.readthedocs.io/en/stable/security.html. A more common way is to use solutions like KNOX to support SSO. 

We need expand this section to more details. (TODO).
