# Pytorch DistributedDataParallel(DDP) Example

## Usage

This is an easy mnist example of how to train a distributed pytorch model using DistributedDataParallel(DDP) method and track the metric and paramater in submarine-sdk.

## How to execute

0. Set up (for a single terminal, only need to do this one time)

```bash
eval $(minikube -p minikube docker-env)
```

1. Build the docker image

```bash
./dev-support/examples/mnist-pytorch/DDP/build.sh
```

2. Submit a post request

```bash
./dev-support/examples/mnist-pytorch/DDP/post.sh
```
