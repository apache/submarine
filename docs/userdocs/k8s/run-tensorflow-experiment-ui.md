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

# Run Tensorflow Experiment on Submarine UI 

## Steps to run Tensorflow Experiment

- Click `+ New Experiment` on the "Experiment" page.  

- Click `Define your experiment`

- Put a name to experiment, like "minst-example" 

- Command: `python /var/tf_mnist/mnist_with_summaries.py --log_dir=/train/log --learning_rate=0.01 --batch_size=150`

- Image you can put; `apache/submarine:tf-mnist-with-summaries-1.0`

- Click `Next Step`

- Choose `Distributed Tensorflow`

- Click `Add new spec` twice to add two new specs (roles) 

- One is Worker, another one is PS, leave rest of the parameters unchanged 

- Click next step, you can review your parameters before submitting the job: 

  

  It should look like below:

  | Name                  | mnist-example-111                                            |      |      |
  | --------------------- | ------------------------------------------------------------ | ---- | ---- |
  | Command               | python /var/tf_mnist/mnist_with_summaries.py --log_dir=/train/log --learning_rate=0.01 --batch_size=150 |      |      |
  | Image                 | apache/submarine:tf-mnist-with-summaries-1.0                 |      |      |
  | Environment Variables |                                                              |      |      |
  | Ps                    | cpu=1,nvidia.com/gpu=0,memory=1024M                          |      |      |
  | Worker                | cpu=1,nvidia.com/gpu=0,memory=1024M                          |      |      |

- Click `Submit` it will be submitted, you can see the new example running in the `Experiment` list, you can get logs, etc. directly from the UI