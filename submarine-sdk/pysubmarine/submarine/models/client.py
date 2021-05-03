"""
 Licensed to the Apache Software Foundation (ASF) under one
 or more contributor license agreements.  See the NOTICE file
 distributed with this work for additional information
 regarding copyright ownership.  The ASF licenses this file
 to you under the Apache License, Version 2.0 (the
 "License"); you may not use this file except in compliance
 with the License.  You may obtain a copy of the License at
 http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing,
 software distributed under the License is distributed on an
 "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 KIND, either express or implied.  See the License for the
 specific language governing permissions and limitations
 under the License.
"""
import os

import mlflow
from mlflow.tracking import MlflowClient

from .constant import (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY,
                       MLFLOW_S3_ENDPOINT_URL, MLFLOW_TRACKING_URI)


class ModelsClient():

    def __init__(self):
        """
        Set up mlflow server connection, including: s3 endpoint, aws, tracking server
        """
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = MLFLOW_S3_ENDPOINT_URL
        os.environ["AWS_ACCESS_KEY_ID"] = AWS_ACCESS_KEY_ID
        os.environ["AWS_SECRET_ACCESS_KEY"] = AWS_SECRET_ACCESS_KEY
        os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI
        self._client = MlflowClient()

    def log_model(self, name, checkpoint):
        mlflow.pytorch.log_model(registered_model_name=name,
                                 pytorch_model=checkpoint,
                                 artifact_path="pytorch-model")

    def load_model(self, name, version):
        model = mlflow.pyfunc.load_model(model_uri=f"models:/{name}/{version}")
        return model

    def update_model(self, name, new_name):
        self._client.rename_registered_model(name=name, new_name=new_name)

    def delete_model(self, name, version):
        self._client.delete_model_version(name=name, version=version)
