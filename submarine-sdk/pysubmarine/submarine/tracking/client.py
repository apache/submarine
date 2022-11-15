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
import json
import os
import re
import tempfile
from typing import Any, Dict, Optional
from datetime import datetime

import submarine
from submarine.artifacts.repository import Repository
from submarine.client.api.serve_client import ServeClient
from submarine.client.utils.api_utils import generate_host
from submarine.entities import Metric, Param
from submarine.exceptions import SubmarineException
from submarine.tracking import utils
from submarine.utils.validation import validate_metric, validate_param

from .constant import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_ENDPOINT_URL


class SubmarineClient:
    """
    Client of an submarine Tracking Server that creates and manages experiments and runs.
    """

    def __init__(
        self,
        db_uri: Optional[str] = None,
        s3_registry_uri: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        host: str = generate_host(),
    ) -> None:
        """
        :param db_uri: Address of local or remote tracking server. If not provided, defaults
                             to the service set by ``submarine.tracking.set_db_uri``. See
                             `Where Runs Get Recorded <../tracking.html#where-runs-get-recorded>`_
                             for more info.
        """
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = s3_registry_uri or S3_ENDPOINT_URL
        os.environ["AWS_ACCESS_KEY_ID"] = aws_access_key_id or AWS_ACCESS_KEY_ID
        os.environ["AWS_SECRET_ACCESS_KEY"] = aws_secret_access_key or AWS_SECRET_ACCESS_KEY
        self.artifact_repo = Repository()
        self.db_uri = db_uri or submarine.get_db_uri()
        self.store = utils.get_tracking_sqlalchemy_store(self.db_uri)
        self.model_registry = utils.get_model_registry_sqlalchemy_store(self.db_uri)
        self.serve_client = ServeClient(host)
        self.experiment_id = utils.get_job_id()

    def log_metric(
        self,
        job_id: str,
        key: str,
        value: float,
        worker_index: str,
        timestamp: datetime = datetime.now(),
        step: int = 0,
    ) -> None:
        """
        Log a metric against the run ID.
        :param job_id: The job name to which the metric should be logged.
        :param key: Metric name.
        :param value: Metric value (float). Note that some special values such
                      as +/- Infinity may be replaced by other values depending on the store. For
                      example, the SQLAlchemy store replaces +/- Inf with max / min float values.
        :param worker_index: Metric worker_index (string).
        :param timestamp: Time when this metric was calculated. Defaults to the current system time.
        :param step: Training step (iteration) at which was the metric calculated. Defaults to 0.
        """
        validate_metric(key, value, timestamp, step)
        metric = Metric(key, value, worker_index, timestamp, step)
        self.store.log_metric(job_id, metric)

    def log_param(self, job_id: str, key: str, value: str, worker_index: str) -> None:
        """
        Log a parameter against the job name. Value is converted to a string.
        :param job_id: The job name to which the parameter should be logged.
        :param key: Parameter name.
        :param value: Parameter value (string).
        :param worker_index: Parameter worker_index (string).
        """
        validate_param(key, value)
        param = Param(key, str(value), worker_index)
        self.store.log_param(job_id, param)

    def save_model(
        self,
        model,
        model_type: str,
        registered_model_name: Optional[str] = None,
        input_dim: Optional[list] = None,
        output_dim: Optional[list] = None,
    ) -> None:
        """
        Save a model into the minio pod or even register a model.
        :param model: Model.
        :param model_type: The type of the model.
        :param registered_model_name: If not None, register model into the model registry with
                                      this name. If None, the model only be saved in minio pod.
        :param input_dim: Save the input dimension of the given model to the description file.
        :param output_dim: Save the output dimension of the given model to the description file.
        """
        pattern = r"[0-9A-Za-z][0-9A-Za-z-_]*[0-9A-Za-z]|[0-9A-Za-z]"
        if registered_model_name and not re.fullmatch(pattern, registered_model_name):
            raise Exception(
                "Registered_model_name must only contains numbers, characters, hyphen and"
                " underscore. Registered_model_name must starts and ends with numbers or"
                " characters."
            )

        model_id = utils.generate_model_id()

        dest_path = self._generate_experiment_artifact_path(f"experiment/{self.experiment_id}")

        # log artifact under the experiment directory
        self._log_artifact(model, dest_path, model_type, model_id, input_dim, output_dim)

        # Register model
        if registered_model_name is not None:
            try:
                self.model_registry.get_registered_model(registered_model_name)
            except SubmarineException:
                self.model_registry.create_registered_model(name=registered_model_name)

            mv = self.model_registry.create_model_version(
                name=registered_model_name,
                id=model_id,
                user_id="",  # TODO(jeff-901): the user id is needed to be specified.
                experiment_id=self.experiment_id,
                model_type=model_type,
            )

            # log artifact under the registry directory
            self._log_artifact(
                model,
                f"registry/{mv.name}-{mv.version}-{model_id}/{mv.name}/{mv.version}",
                model_type,
                model_id,
                input_dim,
                output_dim,
            )

    def _log_artifact(
        self,
        model,
        dest_path: str,
        model_type: str,
        model_id: str,
        input_dim: Optional[list] = None,
        output_dim: Optional[list] = None,
    ):
        """
        Save a model into the minio pod.
        :param model: Model.
        :param dest_path: Destination path of the submarine bucket in the minio pod.
        :param model_type: The type of the model.
        :param model_id: ID of the model.
        :param input_dim: Save the input dimension of the given model to the description file.
        :param output_dim: Save the output dimension of the given model to the description file.
        """
        with tempfile.TemporaryDirectory() as tempdir:
            description: Dict[str, Any] = dict()
            model_save_dir = tempdir
            if not os.path.exists(model_save_dir):
                os.mkdir(model_save_dir)
            if model_type == "pytorch":
                import submarine.models.pytorch

                if input_dim is None or output_dim is None:
                    raise Exception(
                        "Saving pytorch model needs to provide input and output dimension for serving."
                    )
                submarine.models.pytorch.save_model(model, model_save_dir, input_dim)
            elif model_type == "tensorflow":
                import submarine.models.tensorflow

                submarine.models.tensorflow.save_model(model, model_save_dir)
            else:
                raise Exception(f"No valid type of model has been matched to {model_type}")

            # Write description file
            description["id"] = model_id
            if input_dim is not None:
                description["input"] = [
                    {
                        "dims": input_dim,
                    }
                ]
            if output_dim is not None:
                description["output"] = [
                    {
                        "dims": output_dim,
                    }
                ]
            description["model_type"] = model_type
            with open(os.path.join(model_save_dir, "description.json"), "w") as f:
                json.dump(description, f)

            # Log all files into minio
            self.artifact_repo.log_artifacts(dest_path, model_save_dir)

    def _generate_experiment_artifact_path(self, dest_path: str) -> str:
        """
        :param dest_path: destination of current experiment directory
        """
        list_of_subfolder = self.artifact_repo.list_artifact_subfolder(dest_path)
        return (
            os.path.join(dest_path, str(len(list_of_subfolder) + 1))
            if list_of_subfolder
            else os.path.join(dest_path, "1")
        )

    def create_serve(self, model_name: str, model_version: int, async_req: bool = True):
        """
        Create serve of a model through Seldon Core
        :param model_name: Name of a registered model
        :param model_version: Version of a registered model
        """
        return self.serve_client.create_serve(model_name, model_version, async_req)

    def delete_serve(self, model_name: str, model_version: int, async_req: bool = True):
        """
        Delete a serving model
        :param model_name: Name of a registered model
        :param model_version: Version of a registered model
        """
        return self.serve_client.delete_serve(model_name, model_version, async_req=async_req)
