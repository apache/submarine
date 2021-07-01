#!/usr/bin/env bash
#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# Fix submarine-database start failed in kind. https://github.com/kubernetes/minikube/issues/7906
sudo ln -s /etc/apparmor.d/usr.sbin.mysqld /etc/apparmor.d/disable/
sudo apparmor_parser -R /etc/apparmor.d/usr.sbin.mysqld
helm install --wait submarine ./helm-charts/submarine
kubectl get pods
kubectl port-forward svc/submarine-database 3306:3306 &
kubectl port-forward svc/submarine-server 8080:8080 &
kubectl port-forward svc/submarine-minio-service 9000:9000 &
kubectl port-forward svc/submarine-mlflow-service 5001:5000 &