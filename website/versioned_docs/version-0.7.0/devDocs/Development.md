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

# Project Overview
The document [Submarine Quickstart](../gettingStarted/quickstart.md) shows how to deploy the Submarine service to your Kubernetes cluster. The Submarine service consists mainly of nine components, and you can check them with the following command:

```
kubectl get pods -n ${your_namespace}
```

A brief introduction about these components:

1. **tf-operator**: Enable users to run TensorFlow jobs distributedly
2. **pytorch-operator**: Enable users to run PyTorch jobs distributedly
3. **notebook-controller**: Jupyter Notebook controller
4. **submarine-traefik**: Kubernetes Ingress controller
5. **submarine-database**: A MySQL database to store metadata
6. **submarine-minio**: An object store for machine learning artifacts
7. **submarine-mlflow**: A platform for model management
8. **submarine-tensorboard**: A visualization tool for distributed training experiments
9. **submarine-server**: Handle API requests, and submit distributed training experiments to Kubernetes.

# Submarine Development
## Video
* From this [Video](https://youtu.be/32Na2k6Alv4), you will know how to deal with the configuration of Submarine and be able to contribute to it via Github.

## Develop server

### Prerequisites

- JDK 1.8
- Maven 3.3 or later ( < 3.8.1 )
- Docker

### Setting up checkstyle in IDE

Checkstyle plugin may help to detect violations directly from the IDE.

1. Install Checkstyle+IDEA plugin from `Preference` -> `Plugins`
2. Open `Preference` -> `Tools` -> `Checkstyle`
   1. Set Checkstyle version:
      - Checkstyle version: 8.0
   2. Add (+) a new Configuration File
      - Description: Submarine
      - Use a local checkstyle `${SUBMARINE_HOME}/dev-support/maven-config/checkstyle.xml`
3. Open the Checkstyle Tool Window, select the Submarine rule and execute the check

### Testing

- Unit Test

  For each class, there is a corresponding testClass. For example, `SubmarineServerTest` is used for testing `SubmarineServer`. Whenever you add a funtion in classes, you must write a unit test to test it.

- Integration Test: [IntegrationTestK8s.md](./IntegrationTestK8s.md)

### Build from source

- Before building

  1. We assume the developer use **minikube** as a local kubernetes cluster.
  2. Make sure you have **installed the submarine helm-chart** in the cluster.

1. Package the Submarine server into a new jar file

   ```bash
   mvn install -DskipTests
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

3. Delete the server deployment and the operator will create a new one using the new image

   ```bash
   kubectl delete deployment submarine-server
   ```

## Develop workbench

1. Deploy the Submarine

   Follow [Getting Started/Quickstart](../gettingStarted/quickstart.md), and make sure you can connect to `http://localhost:32080` in the browser.

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

4. Frontend E2E test: [IntegrationTestE2E.md](./IntegrationTestE2E.md)

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
## Develop operator

For details, please check out the [README](https://github.com/apache/submarine/blob/master/submarine-cloud-v2/README.md) and [Developer Guide](https://github.com/apache/submarine/blob/master/submarine-cloud-v2/docs/developer-guide.md) on GitHub.

## Develop Submarine Website
Submarine website is built using [Docusaurus 2](https://v2.docusaurus.io/), a modern static website generator.

We store all the website content in markdown format in the `submarine/website/docs`. When committing a new patch to the `submarine` repo, Docusaurus will help us generate the `html` and `javascript` files and push them to  https://github.com/apache/submarine-site/tree/asf-site.

To update the website, click “Edit this page” on the website.

![](https://lh4.googleusercontent.com/gYcKpxbsGAKv2giTRqkxOehPGnuvnhE31WjsAsYhFmACIZF3Wh2ipar7mZ7F_KRwecM-L1J8YJAgNigJsJUjqc-5IXeO2XGxCIcYpP9CdSc3YByuUkjT_Bezby2HHtkBLyE1ZY_F)

### Add a new page
If you want to add a new page to the website, make sure to add the file path to [sidebars.js](https://github.com/apache/submarine/blob/master/website/sidebars.js).

### Installation
We use the yarn package manager to install all dependencies for the website
```console
yarn install
```

### Build
Make sure you can successfully build the website before creating a pull request.
```console
yarn build
```

### Local Development
This command starts a local development server and open up a browser window. Most changes are reflected live without having to restart the server.
```console
yarn start
```
