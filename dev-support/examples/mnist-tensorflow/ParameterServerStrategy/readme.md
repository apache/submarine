# TF ParameterServerStrategy Example (Beta)

## Usage

This is an easy mnist example of how to train a distributed tensorflow model using ParameterServerStrategy and track the metric and paramater in submarine-sdk.

## How to execute

0. Set up (for a single terminal, only need to do this one time)

```bash
eval $(minikube -p minikube docker-env)
```

1. Build the docker image

```bash
./dev-support/examples/mnist-tensorflow/ParameterServerStrategy/build.sh
```

2. Submit a post request

```bash
./dev-support/examples/mnist-tensorflow/ParameterServerStrategy/post.sh
```
