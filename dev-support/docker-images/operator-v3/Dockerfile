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

# Adapted from submarine-cloud-v3/Dockerfile and dev-support/docker-images/operator/Dockerfile

# Build the manager binary
FROM golang:1.17.2 as build-image
MAINTAINER Apache Software Foundation <dev@submarine.apache.org>

ADD tmp/submarine-cloud-v3 /usr/src

WORKDIR /usr/src

# use CGO_ENABLED=0 to support alpine image
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -a -o manager main.go

# use alpine to support shell params
FROM alpine
MAINTAINER Apache Software Foundation <dev@submarine.apache.org>

WORKDIR /usr/src
COPY tmp/submarine-cloud-v3/artifacts/ /usr/src/artifacts/
COPY --from=build-image /usr/src/manager /usr/src/manager

# manager is run as non-root user
# See submarine-cloud-v3/config/manager/manager.yaml
RUN addgroup -g 1000 -S submarine
RUN adduser -u 1000 -S submarine -G submarine
USER submarine

CMD /usr/src/manager -leader-elect=${SUBMARINE_LEADER_ELECT:-false} \
    -istioenable=${SUBMARINE_ISTIO_ENABLE} \
    -submarineateway=${SUBMARINE_ISTIO_SUBMARINE_GATEWAY} \
    -seldonistioenable=${SUBMARINE_ISTIO_SELDON_ENABLE} \
    -seldongateway=${SUBMARINE_ISTIO_SELDON_GATEWAY} \
    -clustertype=${SUBMARINE_CLUSTER_TYPE:kubernetes} \
    -createpsp=${SUBMARINE_POD_SECURITY_POLICY_ENABLE:-true}
