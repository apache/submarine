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

Submarine serve uses istio 1.13.+ (default) and seldon-core 1.10.0 for serving.

## Install

- Install istio first

```bash
# You can download the `istioctl` command by referring to the official website:
# https://istio.io/latest/docs/setup/install/istioctl/
istioctl install -y
```

- Install submarine

```bash
cd submarine
# We add seldon-core to helm, thus we need to update our dependency.
helm dependency update ./helm-charts/submarine
# Install submarine chart
helm install submarine ./helm-charts/submarine --set seldon-core-operator.istio.gateway=submarine/seldon-gateway -n submarine
```

### Uninstall Submarine

```bash
# Uninstall submarine
helm uninstall submarine -n submarine
# Uninstall istio
# https://istio.io/latest/docs/setup/install/istioctl/#uninstall-istio
istioctl uninstall --purge
```
