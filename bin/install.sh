#! /bin/bash
helm install submarine ./helm-charts/submarine
echo ""
kubectl create ns seldon-system
echo ""
helm install seldon-core seldon-core-operator \
    --repo https://storage.googleapis.com/seldon-charts \
    --namespace seldon-system 2> /dev/null