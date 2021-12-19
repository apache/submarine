<!--
  Licensed to the Apache Software Foundation (ASF) under one or more
  contributor license agreements.  See the NOTICE file distributed with
  this work for additional information regarding copyright ownership.
  The ASF licenses this file to You under the Apache License, Version 2.0
  (the "License"); you may not use this file except in compliance with
  the License.  You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
-->

# Submarine Istio

## Pure Istio Setup

Use pure istio gateway.

Note: you have to install [istio](https://istio.io/latest/docs/setup/getting-started/) first.

### Setup

1. Setup Minikube
```bash
minikube start --kubernetes-version=v1.21.7 --cpus=8 --memory=16g
```
2. Setup Istio
```bash
# Only install istiod, no gateway
istioctl install --set profile=demo -y
# Add istio-injection label to allow istio to inject sidecars
kubectl label namespace default istio-injection=enabled
```
3. Install application
```bash
cd submarine-cloud-v2

# Install dependencies
helm install --set dev=true submarine ../helm-charts/submarine

# Start Submarine Operator
make

./submarine-operator

# Install application
kubectl apply -f artifacts/examples/example-submarine.yaml

# Install istio ingress rules
kubectl apply -f ../submarine-istio/submarine-ingress.yaml

# Inspect logs
kubectl logs -n istio-system istio-ingressgateway-8577c57fb6-gd2cr

kubectl logs -c istio-proxy submarine-tensorboard-57c5b64778-wsxqd
```

### Change
- `helm-charts`:
    - Remove traefik
    - Enable Istio in notebook controller (deployment)
- `operator`:
    - Remove traefik resources in artifacts
    - Remove traefik clientset and lister
    - Disable traefik establishment function in operator

### Problems
- 404 error for some routes
- Some resources (service port) seem not conform to Istio format (not sure do they actually affect the application)
- Debugging methods
    - Use `kubectl logs -n istio-system istio-ingressgateway-<xxxx>`
    - Use `kubectl logs -c istio-proxy submarine-<pod>`
    - Use `istioctl analyze`

### Test
- Forward port.
```bash
# Forward Port
kubectl port-forward --address 0.0.0.0 service/istio-ingressgateway -n istio-system 32080:80
```
- Install latest version
```
mvn install -DskipTests
```
- Test k8s
```
mvn verify -DskipRat -pl :submarine-test-k8s -Phadoop-2.9 -B
```
- Test e2e
```
mvn verify -DskipRat -pl :submarine-test-e2e -Phadoop-2.9 -B
```
