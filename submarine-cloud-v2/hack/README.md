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

`update-codegen.sh`

This script will use the generators in [k8s.io/code-generator](https://github.com/kubernetes/code-generator) to generate a typed client, informers, listers and deep-copy functions.

The codes will be generated at `github.com/apache/submarine/submarine-cloud-v2/pkg/client/`.

```bash
# You should be under the folder `submarine-cloud-v2`
./hack/update-codegen.sh
```

`verify-codegen.sh`

This script will verify whether your codes are outdated. If your codes are outdated, please regenerate using `update-codegen.sh`.

```bash
# You should be under the folder `submarine-cloud-v2`
./hack/verify-codegen.sh
```

`build-image.sh`

This script is to make developing with submarine-operator more convenient. It will rebuild the image and recreate the deployment, and one can see the result of changes.

```bash
# You should be under the folder `submarine-cloud-v2`
./hack/build_image.sh <image>

image:
    all             build all images
    server          build submarine server
    database        build submarine database
    jupyter         build submarine jupyter-notebook
    jupyter-gpu     build submarine jupyter-notebook-gpu
    mlflow          build submarine mlflow
```

`server-rapid-builder.sh`

This script is to make the building process of submarine-server faster.

```bash
# You should be under the folder `submarine-cloud-v2/hack`
./server-rapid-builder.sh [namespace]
```

`run_frontend_e2e.sh`

This script is to run frontend end-to-end test. 

```bash
# The workbench should run on port 8080
kubectl port-forward --address 0.0.0.0 -n submarine-user-test service/traefik 8080:80

./hack/run_frontend_e2e.sh
```

