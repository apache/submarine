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

# Build the manager binary
FROM golang:1.17.2 as builder
MAINTAINER Apache Software Foundation <dev@submarine.apache.org>

WORKDIR /workspace
# Copy the Go Modules manifests
COPY go.mod go.mod
COPY go.sum go.sum
# cache deps before building and copying source so that we don't need to re-download as much
# and so that source changes don't invalidate our downloaded layer
RUN go mod download

# Copy the go source
COPY main.go main.go
COPY api/ api/
COPY controllers/ controllers/

# Build
# use CGO_ENABLED=0 to support alpine image
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -a -o manager main.go

# use alpine to support shell params
FROM alpine
MAINTAINER Apache Software Foundation <dev@submarine.apache.org>

WORKDIR /usr/src
COPY artifacts/ /usr/src/artifacts/
COPY --from=builder /workspace/manager /usr/src/manager

# manager is run as non-root user
# See config/manager/manager.yaml
RUN addgroup -g 1000 -S submarine
RUN adduser -u 1000 -S submarine -G submarine
USER submarine

# CMD ls
CMD /usr/src/manager -leader-elect=${SUBMARINE_LEADER_ELECT:-false} \
    -submarineateway=${SUBMARINE_ISTIO_SUBMARINE_GATEWAY} \
    -seldongateway=${SUBMARINE_ISTIO_SELDON_GATEWAY} \
    -clustertype=${SUBMARINE_CLUSTER_TYPE:kubernetes} \
    -createpsp=${SUBMARINE_POD_SECURITY_POLICY_ENABLE:-true}
