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

# Submarine Server Guide
This guide covers the deploy and running the training job by submarine server.

## Prepare environment
Submarine runs on **Linux** and **macOS**, requires the Java 1.8.x or higher. We provide the learning and production environment tutorial. For more deployment info see [Deploy Submarine Server on Kubernetes](./setup-kubernetes.md).

## Training
We designed the generic job spec for training job, suggest to read the the job spec before submit job.

### Job Spec
Job Spec as the DSL for submarine server, it consists of library, submitter and tasks. For example:
```yaml
name: "mnist"
librarySpec:
  name: "TensorFlow"
  version: "2.1.0"
  image: "gcr.io/kubeflow-ci/tf-mnist-with-summaries:1.0"
  cmd: "python /var/tf_mnist/mnist_with_summaries.py --log_dir=/train/log --learning_rate=0.01 --batch_size=150"
  envVars:
    ENV_1: "ENV1"
submitterSpec:
  type: "k8s"
  configPath:
  namespace: "submarine"
  kind: "TFJob"
  apiVersion: "kubeflow.org/v1"
taskSpecs:
  Ps:
    name: tensorflow
    replicas: 2
    resources: "cpu=4,memory=2048M,nvidia.com/gpu=1"
  Worker:
    name: tensorflow
    replicas: 2
    resources: "cpu=4,memory=2048M,nvidia.com/gpu=1"
```
or
```json
{
  "name": "mnist",
  "librarySpec": {
    "name": "TensorFlow",
    "version": "2.1.0",
    "image": "gcr.io/kubeflow-ci/tf-mnist-with-summaries:1.0",
    "cmd": "python /var/tf_mnist/mnist_with_summaries.py --log_dir=/train/log --learning_rate=0.01 --batch_size=150",
    "envVars": {
      "ENV_1": "ENV1"
    }
  },
  "submitterSpec": {
    "type": "k8s",
    "configPath": null,
    "namespace": "submarine",
    "kind": "TFJob",
    "apiVersion": "kubeflow.org/v1"
  },
  "taskSpecs": {
    "Ps": {
      "name": "tensorflow",
      "replicas": 2,
      "resources": "cpu=4,memory=2048M,nvidia.com/gpu=1"
    },
    "Worker": {
      "name": "tensorflow",
      "replicas": 2,
      "resources": "cpu=4,memory=2048M,nvidia.com/gpu=1"
    }
  }
}
```

For more info see [here](../design/submarine-server/jobspec.md).

### Submit Job
> Before submit training job, you should make sure you had deployed the [submarine server and tf-operator](./setup-kubernetes.md#setup-submarine).

You can use the Postman post the job to server or use `curl` run following command:
```
curl -H "Content-Type: application/json" --request POST \
--data '{"name":"mnist","librarySpec":{"name":"TensorFlow","version":"2.1.0","image":"gcr.io/kubeflow-ci/tf-mnist-with-summaries:1.0","cmd":"python /var/tf_mnist/mnist_with_summaries.py --log_dir=/train/log --learning_rate=0.01 --batch_size=150","envVars":{"ENV_1":"ENV1"}},"submitterSpec":{"type":"k8s","configPath":null,"namespace":"submarine","kind":"TFJob","apiVersion":"kubeflow.org/v1"},"taskSpecs":{"Ps":{"name":"tensorflow","replicas":1,"resources":"cpu=1,memory=1024M"},"Worker":{"name":"tensorflow","replicas":1,"resources":"cpu=1,memory=1024M"}}}' \
http://127.0.0.1:8080/api/v1/jobs
```

### Verify Jobs
You can run following command to get the submitted job:
```
kubectl get -n submarine tfjob
```

**Output:**
```
NAME    STATE     AGE
mnist   Created   7m6s
```

Also you can find pods which running the jobs, run following command:
```
kubectl get -n submarine pods
```

**Output:**
```
NAME                               READY   STATUS              RESTARTS   AGE
mnist-ps-0                         0/1     ContainerCreating   0          3m47s
mnist-worker-0                     0/1     Pending             0          3m47s
tf-job-operator-74cc6bd6cb-fqd5s   1/1     Running             0          98m
```
