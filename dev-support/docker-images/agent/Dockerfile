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

FROM alpine:3.16.3
MAINTAINER Apache Software Foundation <dev@submarine.apache.org>

# If you are in China, enabling the following two lines of code can speed up the build of the image, but it may cause failure in travis.
# So when submitting the code, please make sure the following 2 lines of code are in a commented state
# RUN echo "http://mirrors.ustc.edu.cn/alpine/v3.10/main" > /etc/apk/repositories
# RUN echo "http://mirrors.ustc.edu.cn/alpine/v3.10/community" >> /etc/apk/repositories

# INSTALL openjdk
RUN apk update && \
    apk add --no-cache openjdk11 tzdata bash tini&& \
    cp /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && \
    echo Asia/Shanghai > /etc/timezone && \
    apk del tzdata && \
    rm -rf /tmp/* /var/cache/apk/*

ENV JAVA_HOME /usr/lib/jvm/java-11-openjdk/jre

# Install Submarine
RUN mkdir "/opt/submarine-agent"
ADD ./tmp/submarine-k8s-agent-*.jar /opt/submarine-agent/submarine-k8s-agent.jar

WORKDIR /opt/submarine-agent

CMD ["java", "-jar", "/opt/submarine-agent/submarine-k8s-agent.jar"]
