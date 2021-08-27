---
title: Project Architecture
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

## 1. Introduction

This document mainly describes the structure of each module of the Submarine project, the development and test description of each module.

## 2. Submarine Project Structure

### 2.1. submarine-client

Provide the CLI interface for submarine user. (Currently only support YARN service)

### 2.2. submarine-cloud-v2

The operator for Submarine application. For details, please see the [README on github](https://github.com/apache/submarine/blob/master/submarine-cloud-v2/README.md).

### 2.3. submarine-commons

Define utility function used in multiple packages, mainly related to hadoop.

### 2.4. submarine-dist

Store the pre-release files.

### 2.5. submarine-sdk

Provide Python SDK for submarine user.

### 2.6. submarine-security

Provide authorization for Apache Spark to talking to Ranger Admin.

### 2.7. submarine-server

Include core server, restful api, and k8s/yarn submitter.

### 2.8. submarine-test

Provide end-to-end and k8s test for submarine.

### 2.9. submarine-workbench

- workbench-server: is a Jetty-based web server service. Workbench-server provides RESTful interface and Websocket interface. The RESTful interface provides workbench-web with management capabilities for databases such as project, department, user, and role.
- workbench-web: is a web front-end service based on Angular.js framework. With workbench-web users can manage Submarine project, department, user, role through browser. You can also use the notebook to develop machine learning algorithms, model release and other lifecycle management.

### 2.10 dev-support

- **mini-submarine**: by using the docker image provided by Submarine, you can
  experience all the functions of Submarine in a single docker environment, while
  mini-submarine also provides developers with a development and testing
  environment, Avoid the hassle of installing and deploying the runtime
  environment.
- **submarine-installer**: submarine-installer is our submarine runtime
  environment installation tool for yarn-3.1+ and above.By using
  submarine-installer, it is easy to install and deploy system services such as
  `docker`, `nvidia-docker`, `nvidia driver`, `ETCD`, `Calico network` etc.
  required by yarn-3.1+.
