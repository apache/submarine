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

# PySubmarine

PySubmarine is aiming to ease the ML engineer's life by providing a set of libraries.

It includes a high-level out-of-box ML library like deepFM, FM, etc.
low-level library to interact with submarine like creating experiment,
tracking experiment metrics, parameters.

## Package setup

- Install latest version of pysubmarine

```bash
git clone https://github.com/apache/submarine.git
cd submarine/submarine-sdk/pysubmarine
pip install .
```

- Install package from pypi

```bash
pip install apache-submarine
```

## Easy-to-use model trainers

- [FM](https://github.com/apache/submarine/tree/master/submarine-sdk/pysubmarine/example/tensorflow/fm)
- [DeepFM](https://github.com/apache/submarine/tree/master/submarine-sdk/pysubmarine/example/tensorflow/deepfm)

## Submarine experiment management

Makes it easy to run distributed or non-distributed TensorFlow, PyTorch experiments on Kubernetes.

- [mnist example](https://github.com/apache/submarine/tree/master/submarine-sdk/pysubmarine/example/submarine_experiment_sdk.ipynb)

## Development

See [Python Development](https://github.com/apache/submarine/blob/master/website/docs/userDocs/submarine-sdk/pysubmarine/development.md) in the documentation subproject.
