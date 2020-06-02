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


# Deploy Submarine On K8s

## Deploy Submarine using Helm Chart (Recommended)

Submarine's Helm Chart will not only deploy Submarine Server, but also deploys TF Operator / PyTorch Operator (which will be used by Submarine Server to run TF/PyTorch jobs on K8s).

### Create images
submarine server
```bash
./dev-support/docker-images/submarine/build.sh
```

submarine database
```bash
./dev-support/docker-images/database/build.sh
```

### install helm
For more info see https://helm.sh/docs/intro/install/

### Deploy Submarine Server, mysql
You can modify some settings in ./helm-charts/submarine/values.yaml
```bash
helm install submarine ./helm-charts/submarine
```

### Delete deployment
```bash
helm delete submarine
```
