# Submarine Istio

## Traefik-Istio 

Using Traefik as edge router but also leverage istiod as service mesh (exclude gateway).

Note: you have to install [istio](https://istio.io/latest/docs/setup/getting-started/) first.
### Setup

1. Setup Minikube
```bash
minikube start --kubernetes-version=v1.21.7 --cpus=8 --memory=16g
```
2. Setup Istio
```bash
# Only install istiod, no gateway
istioctl install --set profile=minimal -y

# Create separate namespace
kubectl create ns submarine-user-test

# Add istio-injection label to allow istio to inject sidecars
kubectl label namespace default istio-injection=enabled
# You can run `istioctl analyze` to find out the installation problem
```
3. Install application
```bash
# Install dependencies
helm install submarine submarine-istio/

# Install application
kubectl apply -f submarine-cloud-v2/artifacts/examples/example-submarine.yaml
```

### Result
**To be noted**: if you deploy in different namespace, the test will fail now.

Default Namespace:

![default-ns](images/default-ns.png) 

Submarine-User-Test Namespace:

![custom-ns](images/custom-ns.png)

### Test
- Forward port.
```bash
# Forward Port
kubectl port-forward --address 0.0.0.0 service/submarine-traefik 8080:80
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

## Pure Istio Setup
TODO

