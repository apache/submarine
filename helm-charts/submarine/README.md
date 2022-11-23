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

# Submarine Operator Deployment Guide

Debug for test
```shell
# lint
helm lint ./helm-charts/submarine
# dry-run command
helm install --dry-run --debug submarine ./helm-charts/submarine -n submarine
# or template command
helm template --debug submarine ./helm-charts/submarine -n submarine
```

Prod/Dev install
```shell
# We have also integrated seldon-core install by helm, thus we need to update our dependency.
helm dependency update ./helm-charts/submarine
# install submarine operator
helm install submarine ./helm-charts/submarine -n submarine
```

