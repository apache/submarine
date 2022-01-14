helm install submarine ./helm-charts/submarine
kubectl apply -f submarine-cloud-v2/artifacts/examples/example-submarine.yaml
helm install monitor prometheus-community/kube-prometheus-stack -f ./helm-charts/submarine/monitor-values.yaml