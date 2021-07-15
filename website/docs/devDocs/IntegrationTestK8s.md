---
title: How to Run Integration K8s Test
---

<!---
  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License. See accompanying LICENSE file.
-->

## Introduction

* The test cases under the directory `test-k8s` are integration tests to ensure the correctness of the Submarine RESTful API.

* You can run these tests either locally or on GitHub Actions.
  * Before running the tests, the minikube (KinD) cluster must be created. 
  * Then, compile and package the submarine project in `submarine-dist` directory for building a docker image. 
  * In addition, the 8080 port in submarine-traefik should be forwarded.

## Run k8s test locally

1. Ensure you have setup the KinD cluster or minikube cluster. If you haven't, follow this [`minikube tutorial`](https://minikube.sigs.k8s.io/docs/start/)

2. Build the submarine from source and upgrade the server pod through this [`guide`](./Development/#build-from-source)

3. Forward port

  ```bash
  kubectl port-forward --address 0.0.0.0 service/submarine-traefik 8080:80
  ```

4. Execute the test command

  ```bash
  mvn verify -DskipRat -pl :submarine-test-k8s -Phadoop-2.9 -B
  ```

## Run k8s test in GitHub Actions
* Each time a code is submitted, GitHub Actions is triggered automatically.
