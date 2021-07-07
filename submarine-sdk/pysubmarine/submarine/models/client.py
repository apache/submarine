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
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from .constant import (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY,
                       MLFLOW_S3_ENDPOINT_URL, MLFLOW_TRACKING_URI)
from .utils import exist_ps, get_job_id, get_worker_index


class ModelsClient():

    def __init__(self, tracking_uri=None, registry_uri=None):
        """
        Set up mlflow server connection, including: s3 endpoint, aws, tracking server
        """
        # if setting url in environment variable,
        # there is no need to set it by MlflowClient() or mlflow.set_tracking_uri() again
        os.environ[
            "MLFLOW_S3_ENDPOINT_URL"] = registry_uri or MLFLOW_S3_ENDPOINT_URL
        os.environ["AWS_ACCESS_KEY_ID"] = AWS_ACCESS_KEY_ID
        os.environ["AWS_SECRET_ACCESS_KEY"] = AWS_SECRET_ACCESS_KEY
        os.environ["MLFLOW_TRACKING_URI"] = tracking_uri or MLFLOW_TRACKING_URI
        self.client = MlflowClient()
        self.type_to_log_model = {
            "pytorch": mlflow.pytorch.log_model,
            "sklearn": mlflow.sklearn.log_model,
            "tensorflow": mlflow.tensorflow.log_model,
            "keras": mlflow.keras.log_model
        }

    def start(self):
        """
        1. Start a new Mlflow run
        2. Direct the logging of the artifacts and metadata
            to the Run named "worker_i" under Experiment "job_id"
        3. If in distributed training, worker and job id would be parsed from environment variable
        4. If in local traning, worker and job id will be generated.
        :return: Active Run
        """
        experiment_name = get_job_id()
        run_name = get_worker_index()
        experiment_id = self._get_or_create_experiment(experiment_name)
        return mlflow.start_run(run_name=run_name, experiment_id=experiment_id)

    def log_param(self, key, value):
        mlflow.log_param(key, value)

    def log_params(self, params):
        mlflow.log_params(params)

    def log_metric(self, key, value, step=None):
        mlflow.log_metric(key, value, step)

    def log_metrics(self, metrics, step=None):
        mlflow.log_metrics(metrics, step)

    def load_model(self, name, version):
        model = mlflow.pyfunc.load_model(model_uri=f"models:/{name}/{version}")
        return model

    def update_model(self, name, new_name):
        self.client.rename_registered_model(name=name, new_name=new_name)

    def delete_model(self, name, version):
        self.client.delete_model_version(name=name, version=version)

    def save_model(self,
                   model_type,
                   model,
                   artifact_path,
                   registered_model_name=None):
        run_name = get_worker_index()
        if exist_ps():
            # TODO for Tensorflow ParameterServer strategy
            return
        elif run_name == "worker-0":
            if model_type in self.type_to_log_model:
                self.type_to_log_model[model_type](
                    model,
                    artifact_path,
                    registered_model_name=registered_model_name)
            else:
                raise MlflowException("No valid type of model has been matched")

    def _get_or_create_experiment(self, experiment_name):
        """
        Return the id of experiment.
        If non-exist, create one. Otherwise, return the existing one.
        :return: Experiment id
        """
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:  # if not found
                raise MlflowException("No valid experiment has been found")
            return experiment.experiment_id  # if found
        except MlflowException:
            experiment = mlflow.create_experiment(name=experiment_name)
            return experiment
