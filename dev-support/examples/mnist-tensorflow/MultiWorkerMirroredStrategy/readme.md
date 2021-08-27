# TF MultiWorkerMirroredStrategy Example

## Usage

This is an easy mnist example of how to train a distributed tensorflow model using MultiWorkerMirroredStrategy and track the metric in submarine-sdk.

## How to execute

0. Set up (for a single terminal, only need to do this one time)

```bash
eval $(minikube -p minikube docker-env)
```

1. Build the docker image

```bash
./dev-support/examples/mnist-tensorflow/MultiWorkerMirroredStrategy/build.sh
```

2. Submit a post request

```bash
./dev-support/examples/mnist-tensorflow/MultiWorkerMirroredStrategy/post.sh
```
