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

FROM python:3.7.0-slim

RUN apt-get update && apt-get -y install --no-install-recommends default-libmysqlclient-dev libpq-dev build-essential wget

RUN pip install greenlet==1.1.3 mlflow==1.15.0 sqlalchemy==1.4.11 boto3==1.17.58 pymysql==0.9.3

COPY start.sh /usr/local/bin

WORKDIR /usr/local/bin

ENV MLFLOW_S3_ENDPOINT_URL http://submarine-minio-service:9000
ENV AWS_ACCESS_KEY_ID submarine_minio
ENV AWS_SECRET_ACCESS_KEY submarine_minio
ENV BACKEND_URI mysql+pymysql://mlflow:password@submarine-database:3306/mlflowdb


EXPOSE 5000

CMD ./start.sh