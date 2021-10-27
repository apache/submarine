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

FROM ubuntu:18.04
MAINTAINER Apache Software Foundation <dev@submarine.apache.org>

WORKDIR /usr/src

RUN apt-get update &&\
    apt-get install -y wget vim git curl

ENV GOROOT="/usr/local/go"
ENV GOPATH=$HOME/gocode
ENV GOBIN=$GOPATH/bin
ENV PATH=$PATH:$GOPATH:$GOBIN:$GOROOT/bin

RUN wget https://dl.google.com/go/go1.16.2.linux-amd64.tar.gz &&\
    tar -C /usr/local -xzf go1.16.2.linux-amd64.tar.gz

RUN curl -LO https://dl.k8s.io/release/v1.14.2/bin/linux/amd64/kubectl &&\
    install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl &&\
    kubectl version --client

ADD submarine-operator /usr/src
COPY ["./artifacts", "/usr/src/artifacts"]

CMD ["/usr/src/submarine-operator", "-incluster=true"]
