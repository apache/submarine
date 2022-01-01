# Tracking Example

## Usage
This is an easy example of how to track the metric and paramater in submarine-sdk. Basically, the sdk will detect which experiment and worker-id are you at, and log your data to the corresponding place. 

For example, you start an experiment with 3 workers. Suppose the experiment is assigned with an ID `experiment_12345678`, and the operator launches 3 pods with worker_id `worker-0`, `worker-1` and `worker-2` respectively. 

The logging of `worker-i` will be directed to `experiment_12345678` / `worker-i` in the submarine server 

## How to execute

1. Build the docker image

```bash
./dev-support/examples/tracking/build.sh
```

2. Submit a post request

```bash
./dev-support/examples/tracking/post.sh
```
