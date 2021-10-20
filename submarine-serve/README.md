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

# Submarine Serve

Submarine serve uses istio 1.6.8 and seldon-core 1.10.0 for serving.

## Install

- Install submarine with istio and seldon-core

```bash
cd submarine
./submarine-serve/installation/install.sh
```

### Uninstall Submarine

```bash
helm delete submarine
kubectl delete ns istio-system
kubectl delete ns seldon-system
```
