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

FROM alpine:3.10
MAINTAINER Apache Software Foundation <dev@submarine.apache.org>

# INSTALL openjdk
RUN apk update && \
    apk add --no-cache openjdk8 tzdata bash tini && \
    cp /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && \
    echo Asia/Shanghai > /etc/timezone && \
    apk del tzdata && \
    rm -rf /tmp/* /var/cache/apk/*

ENV JAVA_HOME /usr/lib/jvm/java-1.8-openjdk/jre

# Install Submarine
ADD ./tmp/submarine-dist-*.tar.gz /opt/
RUN ln -s /opt/submarine-dist-* "/opt/submarine-current"
ADD ./tmp/submarine-site.xml "/opt/submarine-current/conf/"
ADD ./tmp/submarine.sh "/opt/submarine-current/bin/"
ADD ./tmp/mysql-connector-java-5.1.39.jar "/opt/submarine-current/lib/"

WORKDIR /opt/submarine-current

# Submarine port
EXPOSE 8080

ENTRYPOINT ["/sbin/tini", "--"]

CMD ["/bin/bash", "-c", "/opt/submarine-current/bin/submarine.sh start"]
