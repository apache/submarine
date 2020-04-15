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

# Submarine Client API Example
This example shows how you could use Submarine Client API to 
manage submarine tasks

## Prerequisites
- [Deploy Submarine Server on Kubernetes](https://github.com/apache/submarine/blob/master/docs/submarine-server/setup-kubernetes.md)
- [Deploy Tensorflow Operator on Kubernetes](https://github.com/apache/submarine/blob/master/docs/submarine-server/ml-frameworks/tensorflow.md)

#### Submit Job
1. Create a job description for submarine client. e.g.[mnist.json](./mnist.json)

2. Create Submarine job client
```python
from submarine.job import SubmarineJobClient
client = SubmarineJobClient('localhost', 8080)
```
3. Submit job
```python
response = client.submit_job('mnist.json')
```
#### Delete job
```python
response = client.delete_job('job_1586791302310_0005')
```
