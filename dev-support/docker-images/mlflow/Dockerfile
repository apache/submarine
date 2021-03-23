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

RUN pip install mlflow

RUN mkdir /submarine-mlflow

WORKDIR /submarine-mlflow

RUN apt-get update && apt-get -y install --no-install-recommends default-libmysqlclient-dev libpq-dev build-essential sqlite3
    
RUN pip install \
    mlflow \
    sqlalchemy \
    boto3

RUN sqlite3 store.db

EXPOSE 5000

ENV BACKEND_URI sqlite:///store.db
ENV ARTIFACT_ROOT /submarine-mlflow/mlflow-artifacts

CMD mlflow server \
    --host 0.0.0.0 \
    --backend-store-uri ${BACKEND_URI} \
    --default-artifact-root ${ARTIFACT_ROOT} \
    --static-prefix "/mlflow"