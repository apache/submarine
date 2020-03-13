#!/bin/bash
kubectl apply -f ./namespace.yaml
sleep 5
# there're two deployment yaml file. One for K8s version prior to 1.16. One for 1.16 due to API incompatability
kubectl apply -f ./
