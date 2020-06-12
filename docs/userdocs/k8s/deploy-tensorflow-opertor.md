<!--
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
-->

# Deploy TensorFlow Operator on Kubernetes

## TFJob
We support TensorFlow job on kubernetes by using the tf-operator as a runtime. For more info about tf-operator see [here](https://github.com/kubeflow/tf-operator).

### Deploy tf-operator
> If you don't have the `submarine` namespace on your K8s cluster, you should create it first. Run command: `kubectl create namespace submarine`

Running the follow commands:
```
kubectl apply -f ./dev-support/k8s/tfjob/crd.yaml
kubectl kustomize ./dev-support/k8s/tfjob/operator | kubectl apply -f -
```

> Since K8s 1.14, Kubectl also supports the management of Kubernetes objects using a kustomization file. For more info see [kustomization](https://kubernetes.io/docs/tasks/manage-kubernetes-objects/kustomization/)

Default namespace is `submarine`, if you want to modify the namespace, please modify `./dev-support/k8s/tfjob/operator/kustomization.yaml`, such as modify `${NAMESPACE}` as below:
```yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
namespace: ${NAMESPACE}
resources:
- cluster-role-binding.yaml
- cluster-role.yaml
- deployment.yaml
- service-account.yaml
- service.yaml
commonLabels:
  kustomize.component: tf-job-operator
images:
- name: gcr.io/kubeflow-images-public/tf_operator
  newName: gcr.io/kubeflow-images-public/tf_operator
  newTag: v0.7.0
```

