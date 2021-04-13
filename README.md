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

<div align="center">

![Colored_logo_with_text](website/docs/assets/color_logo_with_text.png)

[![Build Status](https://travis-ci.org/apache/submarine.svg?branch=master)](https://travis-ci.org/apache/submarine) [![License](https://img.shields.io/badge/license-Apache%202-4EB1BA.svg)](https://www.apache.org/licenses/LICENSE-2.0.html)  [![HitCount](http://hits.dwyl.com/apache/submarine.svg)](http://hits.dwyl.io/apache/submarine) [![PyPI version](https://badge.fury.io/py/apache-submarine.svg)](https://badge.fury.io/py/apache-submarine)

</div>
# What is Apache Submarine?

**Apache Submarine** (Submarine for short) is an **End-to-End Machine Learning Platform** to allow data scientists to create end-to-end machine learning workflows. On **Submarine**, data scientists can finish each stage in the ML model lifecycle, including data exploration, data pipeline creation, model training, serving, and monitoring.

## Why Submarine?

Some open-source and commercial projects are trying to build an end-to-end ML platform. What's the vision of Submarine?

### Problems

1) Many platforms lack easy-to-use user interfaces (API, SDK, and IDE, etc.)
2) In the same company, data scientists in different teams usually spend much time on developments of existing feature sets and models.
3) Data scientists put emphasis on domain-specific tasks (e.g. Click-Through-Rate), but they need to implement their models from scratch with SDKs provided by existing platforms.
4) Many platforms lack a unified workbench to manage each component in the ML lifecycle.

_Theodore Levitt_ once said:

```
“People don’t want to buy a quarter-inch drill. They want a quarter-inch hole.”
```

### Goals of Submarine

#### Model Training (Experiment)
- Run/Track distributed training `experiment` on prem or cloud via easy-to-use UI/API/SDK.
- Easy for data scientists to manage versions of `experiment` and dependencies of `environment`
- Support popular machine learning frameworks, including **TensorFlow**, **PyTorch**, **Horovod**, and **MXNet**
- Provide pre-defined **template** for data scientists to implement domain-specific tasks easily (e.g. using DeepFM template to build a CTR prediction model)
- Support many compute resources (e.g. CPU and GPU, etc.)
- Support **Kubernetes** and **YARN**
- Pipeline is also on the backlog, we will look into pipeline for training in the future.

#### Notebook Service

- Submarine aims to provide a notebook service (e.g. Jupyter notebook) which allows users to manage notebook instances running on the cluster.

#### Model Management (Serving/versioning/monitoring, etc.)

- Model management for model-serving/versioning/monitoring is on the roadmap.

## Easy-to-use User Interface

As mentioned above, Submarine attempts to provide **Data-Scientist-friendly** UI to make data scientists have a good user experience. Here're some examples.

### Example: Submit a distributed Tensorflow experiment via Submarine Python SDK

#### Run a Tensorflow Mnist experiment
```python

# New a submarine client of the submarine server
submarine_client = submarine.ExperimentClient(host='http://localhost:8080')

# The experiment's environment, could be Docker image or Conda environment based
environment = EnvironmentSpec(image='apache/submarine:tf-dist-mnist-test-1.0')

# Specify the experiment's name, framework it's using, namespace it will run in,
# the entry point. It can also accept environment variables. etc.
# For PyTorch job, the framework should be 'Pytorch'.
experiment_meta = ExperimentMeta(name='mnist-dist',
                                 namespace='default',
                                 framework='Tensorflow',
                                 cmd='python /var/tf_dist_mnist/dist_mnist.py --train_steps=100')
# 1 PS task of 2 cpu, 1GB
ps_spec = ExperimentTaskSpec(resources='cpu=2,memory=1024M',
                             replicas=1)
# 1 Worker task
worker_spec = ExperimentTaskSpec(resources='cpu=2,memory=1024M',
                                 replicas=1)

# Wrap up the meta, environment and task specs into an experiment.
# For PyTorch job, the specs would be "Master" and "Worker".
experiment_spec = ExperimentSpec(meta=experiment_meta,
                                 environment=environment,
                                 spec={'Ps':ps_spec, 'Worker': worker_spec})

# Submit the experiment to submarine server
experiment = submarine_client.create_experiment(experiment_spec=experiment_spec)

# Get the experiment ID
id = experiment['experimentId']

```

#### Query a specific experiment
```python
submarine_client.get_experiment(id)
```

#### Wait for finish

```python
submarine_client.wait_for_finish(id)
```

#### Get the experiment's log
```python
submarine_client.get_log(id)
```

#### Get all running experiment
```python
submarine_client.list_experiments(status='running')
```


For a quick-start, see [Submarine On K8s](https://submarine.apache.org/docs/adminDocs/k8s/README)


### Example: Submit a pre-defined experiment template job

### Example: Submit an experiment via Submarine UI

(Available on 0.6.0, see Roadmap)

## Architecture, Design and requirements

If you want to know more about Submarine's architecture, components, requirements and design doc, they can be found on [Architecture-and-requirement](https://submarine.apache.org/docs/designDocs/architecture-and-requirements)

Detailed design documentation, implementation notes can be found at: [Implementation notes](https://submarine.apache.org/docs/designDocs/implementation-notes)

## Apache Submarine Community

Read the [Apache Submarine Community Guide](https://submarine.apache.org/docs/community/README)

How to contribute [Contributing Guide](https://submarine.apache.org/docs/community/contributing)


Issue Tracking: https://issues.apache.org/jira/projects/SUBMARINE

## User Document


See [User Guide Home Page](https://submarine.apache.org/docs/)

## Developer Document

See [Developer Guide Home Page](https://submarine.apache.org/docs/devDocs/Development/)

## Roadmap

What to know more about what's coming for Submarine? Please check the roadmap out: https://cwiki.apache.org/confluence/display/SUBMARINE/Roadmap

## License

The Apache Submarine project is licensed under the Apache 2.0 License. See the [LICENSE](./LICENSE) file for details.
