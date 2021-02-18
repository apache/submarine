---
title: Development Guide
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

From [Getting Started/Submarine Local Deployment](../gettingStarted/localDeployment.md), you already know that Submarine is installed and uninstalled by Helm. As you can see by `kubectl get pods`, there are six major components in Submarine, including `notebook-controller`, `pytorch-operator`, `submarine-database`, `submarine-server`, `submarine-traefik` and `tf-job-operator`. They are launched as pods in kubernetes from the corresponding docker images.

Some of the components are borrowed from other projects (kubeflow, traefik), including `notebook-controller`, `pytorch-operator`, `submarine-traefik` and `tf-job-operator`. The rest of them are built by ourselves, including `submarine-database` and `submarine-server`.

The purpose of the components are as the following:

1. `tf-job-operator`: manage the operation of tensorflow jobs
2. `pytorch-operator`: manage the operation of pytorch jobs
3. `notebook-controller`: manage the operation of notebook instances
4. `submarine-traefik`: manage the ingress service

5. `submarine-database`: store metadata in mysql database
6. `submarine-server`: handle api request, submit job to container orchestration, and connect with database.

In this document, we only focus on the last two components. You can learn how to develop server, database, and workbench here.

## Develop server

### Prerequisites

- JDK 1.8
- Maven 3.3 or later ( 3.6.2 is known to fail, see SUBMARINE-273 )
- Docker

### Setting up checkstyle in IDE

Checkstyle plugin may help to detect violations directly from the IDE.

1. Install Checkstyle+IDEA plugin from Preference -> Plugins
2. Open Preference -> Tools -> Checkstyle ->
    1. Set Checkstyle version:
        - Checkstyle version: 8.0
    2. Add (+) a new Configuration File
        - Description: Submarine
        - Use a local checkstyle ${SUBMARINE_HOME}/dev-support/maven-config/checkstyle.xml
3. Open the Checkstyle Tool Window, select the Submarine rule and execute the check

### Testing

- Unit Test

    For each class, there is a corresponding testClass. For example, `SubmarineServerTest` is used for testing `SubmarineServer`. Whenever you add a funtion in classes, you must write a unit test to test it.

- Integration Test

    See [IntegrationTest.md](./IntegrationTest.md)

### Build from source

- Before building

    1. We assume the developer use **minikube** as a local kubernetes cluster.
    2. Make sure you have **installed the submarine helm-chart** in the cluster.

1. Package the Submarine server into a new jar file

    ```bash
    mvn package -DskipTests
    ```

2. Build the new server docker image in minikube

    ```bash
    # switch to minikube docker daemon to build image directly in minikube
    eval $(minikube docker-env)

    # run docker build
    ./dev-support/docker-images/submarine/build.sh

    # exit minikube docker daemon
    eval $(minikube docker-env -u)
    ```

3. Update server pod

    ```bash
    helm upgrade --set submarine.server.dev=true submarine ./helm-charts/submarine
    ```

    Set `submarine.server.dev` to `true`, enabling the server pod to be launched with the new docker image.

## Develop workbench

1. Deploy the Submarine

    Follow [Getting Started/Submarine Local Deployment](../gettingStarted/localDeployment.md), and make sure you can connect to `http://localhost:32080` in the browser.

2. Install the dependencies

    ```bash
    cd submarine-workbench/workbench-web
    npm install
    ```

3. Run the workbench based on proxy server

    ```bash
    npm run start
    ```

    1. The request sent to `http://localhost:4200` will be redirected to `http://localhost:32080`.
    2. Open `http://localhost:4200` in browser to see the real-time change of workbench.

## Develop database

1. Build the docker image

    ```bash
    # switch to minikube docker daemon to build image directly in minikube
    eval $(minikube docker-env)

    # run docker build
    ./dev-support/docker-images/database/build.sh

    # exit minikube docker daemon
    eval $(minikube docker-env -u)
    ```

2. Deploy new pods in the cluster

    ```bash
    helm upgrade --set submarine.database.dev=true submarine ./helm-charts/submarine
    ```