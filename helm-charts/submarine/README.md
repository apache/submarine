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

# Submarine Operator Deployment Guide

- Installs the Cloud Native Machine Learning Platform [Apache Submarine](https://submarine.apache.org/)

## Debugging the Chart

To debug the chart with the release name `submarine`:

```shell
# lint
helm lint ./helm-charts/submarine
# dry-run command
helm install --dry-run --debug submarine ./helm-charts/submarine -n submarine
# or template command
helm template --debug submarine ./helm-charts/submarine -n submarine
```

## Installing the Chart

To install the chart with the release name `submarine`:

```shell
# We have also integrated seldon-core install by helm, thus we need to update our dependency.
helm dependency update ./helm-charts/submarine
# install submarine operator
helm install submarine ./helm-charts/submarine -n submarine
```

## Upgrading the Chart

To upgrade the chart with the release name `submarine`:

```shell
helm upgrade submarine ./helm-charts/submarine -n submarine
```

## Uninstalling the Chart

To uninstall/delete the `subamrine` deployment:

```shell
helm uninstall subamrine -n submarine
```

## Upgrading an existing Release to a new major version

A major chart version change (like v0.8.0 -> v1.0.0) indicates that there is an
incompatible breaking change needing manual actions.

### To 0.8.0

This version requires Helm >= 3.1.0.  
This version is a major change, we migrated `traefik` to `istio` and upgraded the `operator`. You need to backup the database and redeploy.

## Configuration

The following table lists the configurable parameters of the MySQL chart and their default values.

| Parameter                                    | Description                                                                                                                                   | Default                                    |
| -------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------ |
| `name`                                       | Submarine operator deployment name.                                                                                                           | `submarine-operator`                       |
| `replicas`                                   | Number of operators                                                                                                                           | `1`                                        |
| `image`                                      | Submarine operator deployment image                                                                                                           | `apache/submarine:operator-0.8.0-SNAPSHOT` |
| `imagePullSecrets`                           | Image pull secrets                                                                                                                            | `[]`                                       |
| `dev`                                        | Tell helm to install submarine-operator or not in dev mode                                                                                    | `false`                                    |
| `storageClass.reclaimPolicy`                 | Determine the action after the persistent volume is released                                                                                  | `Delete`                                   |
| `storageClass.volumeBindingMode`             | Control when volume binding and dynamically provisioning should occur                                                                         | `Immediate`                                |
| `storageClass.provisioner`                   | Determine what volume plugin is used for provisioning PVs                                                                                     | `k8s.io/minikube-hostpath`                 |
| `storageClass.parameters`                    | Describe volumes belonging to the storage class                                                                                               | `{}`                                       |
| `clusterType`                                | k8s cluster type. can be: kubernetes or openshift                                                                                             | `kubernetes`                               |
| `podSecurityPolicy.create`                   | pecifies whether a PodSecurityPolicy should be created, this configuration enables the database/minio/server to set securityContext.runAsUser | `true`                                     |
| `istio.enabled`                              | Use istio to expose the service                                                                                                               | `true`                                     |
| `istio.gatewaySelector`                      | Gateway label selector                                                                                                                        | `istio: ingressgateway`                    |
| `training-operator.enabled`                  | If we need to deploye a kubeflow training operator in this helm                                                                               | `true`                                     |
| `training-operator.image.pullPolicy`         | Training operator image pull policy                                                                                                           | `IfNotPresent`                             |
| `training-operator.image.registry`           | Training operator image registry                                                                                                              | `public.ecr.aws`                           |
| `training-operator.image.repository`         | Training operator image repository                                                                                                            | `j1r0q0g6/training/training-operator`      |
| `training-operator.image.tag`                | Training operator image tag                                                                                                                   | `760ac1171dd30039a7363ffa03c77454bd714da5` |
| `training-operator.image.imagePullSecrets`   | Training operator image pull Secrets                                                                                                          | `[]`                                       |
| `notebook-controller.enabled`                | If we need to deploye a kubeflow notebook controller in this helm                                                                             | `true`                                     |
| `notebook-controller.image.pullPolicy`       | Notebook controller image pull policy                                                                                                         | `IfNotPresent`                             |
| `notebook-controller.image.registry`         | Notebook controller image registry                                                                                                            | `docker.io`                                |
| `notebook-controller.image.repository`       | Notebook controller image repository                                                                                                          | `apache/submarine`                         |
| `notebook-controller.image.tag`              | Notebook controller image tag                                                                                                                 | `notebook-controller-v1.4`                 |
| `notebook-controller.image.imagePullSecrets` | Notebook controller image pull Secrets                                                                                                        | `[]`                                       |
| `seldon-core-operator.enabled`               | If we need to deploye a seldon core operator in this helm                                                                                     | `true`                                     |
