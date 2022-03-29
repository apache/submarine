---
title: Quickstart
---

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

This document gives you a quick view on the basic usage of Submarine platform. You can finish each step of ML model lifecycle on the platform without messing up with the troublesome environment problems.

## Installation

### Prepare a Kubernetes cluster

1. Prerequisite

- Check [dependency page](../devDocs/Dependencies.md) for the compatible version
- [kubectl](https://kubernetes.io/docs/tasks/tools/install-kubectl/)
- [helm](https://helm.sh/docs/intro/install/) (Helm v3 is minimum requirement.)
- [minikube](https://minikube.sigs.k8s.io/docs/start/).
- [istioctl](https://istio.io/latest/docs/setup/getting-started/#download)

2. Start minikube cluster and install Istio

```
minikube start --vm-driver=docker --cpus 8 --memory 8192 --kubernetes-version v1.21.2
istioctl install -y
# Or if you want to support Pod Security Policy (https://minikube.sigs.k8s.io/docs/tutorials/using_psp), you can use the following command to start cluster
minikube start --extra-config=apiserver.enable-admission-plugins=PodSecurityPolicy --addons=pod-security-policy --vm-driver=docker --cpus 8 --memory 4096 --kubernetes-version v1.21.2
```

### Launch submarine in the cluster

1. Clone the project

```
git clone https://github.com/apache/submarine.git
cd submarine
```

2. Create necessary namespaces

```
kubectl create namespace submarine
kubectl create namespace submarine-user-test
kubectl label namespace submarine istio-injection=enabled
kubectl label namespace submarine-user-test istio-injection=enabled
```

3. Install the submarine operator and dependencies by helm chart

```
helm install submarine ./helm-charts/submarine -n submarine
```

4. Create a Submarine custom resource and the operator will create the submarine server, database, etc. for us.

```
kubectl apply -f submarine-cloud-v2/artifacts/examples/example-submarine.yaml -n submarine-user-test
```

### Ensure submarine is ready

```
$ kubectl get pods -n submarine
NAME                                              READY   STATUS    RESTARTS   AGE
notebook-controller-deployment-66d85984bf-x562z   1/1     Running   0          7h7m
pytorch-operator-7d778f4859-g7xph                 2/2     Running   0          7h7m
tf-job-operator-7d895bf77c-75n72                  2/2     Running   0          7h7m

$ kubectl get pods -n submarine-user-test
NAME                                     READY   STATUS    RESTARTS   AGE
submarine-database-bdcb77549-rq2ds       2/2     Running   0          7h6m
submarine-minio-686b8777ff-zg4d2         2/2     Running   0          7h6m
submarine-mlflow-68c5559dcb-lkq4g        2/2     Running   0          7h6m
submarine-server-7c6d7bcfd8-5p42w        2/2     Running   0          9m33s
submarine-tensorboard-57c5b64778-t4lww   2/2     Running   0          7h6m
```

### Connect to workbench

1. Exposing service

```
kubectl port-forward --address 0.0.0.0 -n istio-system service/istio-ingressgateway 32080:80
```

2. View workbench

Go to `http://0.0.0.0:32080`
![](/img/quickstart-worbench.png)

## Example: Submit a mnist distributed example

We put the code of this example [here](https://github.com/apache/submarine/tree/master/dev-support/examples/quickstart). `train.py` is our training script, and `build.sh` is the script to build a docker image.

### 1. Write a python script for distributed training

Take a simple mnist tensorflow script as an example. We choose `MultiWorkerMirroredStrategy` as our distributed strategy.

```python
"""
./dev-support/examples/quickstart/train.py
Reference: https://github.com/kubeflow/tf-operator/blob/master/examples/v1/distribution_strategy/keras-API/multi_worker_strategy-with-keras.py
"""

import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.keras import layers, models
import submarine

def make_datasets_unbatched():
  BUFFER_SIZE = 10000

  # Scaling MNIST data from (0, 255] to (0., 1.]
  def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255
    return image, label

  datasets, _ = tfds.load(name='mnist', with_info=True, as_supervised=True)

  return datasets['train'].map(scale).cache().shuffle(BUFFER_SIZE)


def build_and_compile_cnn_model():
  model = models.Sequential()
  model.add(
      layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
  model.add(layers.MaxPooling2D((2, 2)))
  model.add(layers.Conv2D(64, (3, 3), activation='relu'))
  model.add(layers.MaxPooling2D((2, 2)))
  model.add(layers.Conv2D(64, (3, 3), activation='relu'))
  model.add(layers.Flatten())
  model.add(layers.Dense(64, activation='relu'))
  model.add(layers.Dense(10, activation='softmax'))

  model.summary()

  model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

  return model

def main():
  strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(
      communication=tf.distribute.experimental.CollectiveCommunication.AUTO)

  BATCH_SIZE_PER_REPLICA = 4
  BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

  with strategy.scope():
    ds_train = make_datasets_unbatched().batch(BATCH_SIZE).repeat()
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = \
        tf.data.experimental.AutoShardPolicy.DATA
    ds_train = ds_train.with_options(options)
    # Model building/compiling need to be within `strategy.scope()`.
    multi_worker_model = build_and_compile_cnn_model()

  class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
      # monitor the loss and accuracy
      print(logs)
      submarine.log_metrics({"loss": logs["loss"], "accuracy": logs["accuracy"]}, epoch)

  multi_worker_model.fit(ds_train, epochs=10, steps_per_epoch=70, callbacks=[MyCallback()])


if __name__ == '__main__':
  main()
```

### 2. Prepare an environment compatible with the training

Build a docker image equipped with the requirement of the environment.

```bash
eval $(minikube docker-env)
./dev-support/examples/quickstart/build.sh
```

### 3. Submit the experiment

1. Open submarine workbench and click `+ New Experiment`
2. Choose `Define your experiment`
3. Fill the form accordingly. Here we set 3 workers.

   1. Step 1
      ![](/img/quickstart-submit-1-0-7-0.png)
   2. Step 2
      ![](/img/quickstart-submit-2-0-7-0.png)
   3. Step 3
      ![](/img/quickstart-submit-3-0-7-0.png)
   4. The experiment is successfully submitted
      ![](/img/quickstart-submit-4-0-7-0.png)

### 4. Monitor the process

1. In our code, we use `submarine` from `submarine-sdk` to record the metrics. To see the result, click corresponding experiment with name `mnist-example` in the workbench.
2. To see the metrics of each worker, you can select a worker from the left top list.

![](/img/quickstart-ui-0-7-0.png)

### 5. Serve the model (In development)
