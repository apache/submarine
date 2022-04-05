# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

FROM golang:1.16.2 AS build-image
MAINTAINER Apache Software Foundation <dev@submarine.apache.org>

ADD tmp/submarine-cloud-v2 /usr/src

WORKDIR /usr/src

RUN GOOS=linux go build -o submarine-operator

FROM gcr.io/distroless/base-debian10
ADD tmp/submarine-cloud-v2/artifacts /usr/src/artifacts
WORKDIR /usr/src
COPY --from=build-image /usr/src/submarine-operator /usr/src/submarine-operator

CMD ["/usr/src/submarine-operator", "-incluster=true", "-clustertype=${SUBMARINE_CLUSTER_TYPE}", "-createpsp=${SUBMARINE_POD_SECURITY_POLICY_ENABLE}"]
