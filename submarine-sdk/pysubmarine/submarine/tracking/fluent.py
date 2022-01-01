# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements. See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Internal module implementing the fluent API, allowing management of an active
Submarine run. This module is exposed to users at the top-level :py:mod:`submarine` module.
"""
from __future__ import print_function

import logging
from datetime import datetime

from submarine.tracking.client import SubmarineClient
from submarine.tracking.utils import get_job_id, get_worker_index

_RUN_ID_ENV_VAR = "SUBMARINE_RUN_ID"

_logger = logging.getLogger(__name__)


def log_param(key, value):
    """
    Log a parameter under the current run, creating a run if necessary.
    :param key: Parameter name (string)
    :param value: Parameter value (string, but will be string-field if not)
    """
    job_id = get_job_id()
    worker_index = get_worker_index()
    SubmarineClient().log_param(job_id, key, value, worker_index)


def log_metric(key, value, step=None):
    """
    Log a metric under the current run, creating a run if necessary.
    :param key: Metric name (string).
    :param value: Metric value (float). Note that some special values such as +/- Infinity may be
                  replaced by other values depending on the store. For example, sFor example, the
                  SQLAlchemy store replaces +/- Inf with max / min float values.
    :param step: Metric step (int). Defaults to zero if unspecified.
    """
    job_id = get_job_id()
    worker_index = get_worker_index()
    SubmarineClient().log_metric(job_id, key, value, worker_index, datetime.now(), step or 0)


def save_model(
    model_type: str,
    model,
    artifact_path: str,
    registered_model_name: str = None,
    input_dim: list = None,
    output_dim: list = None,
):
    """
    Save a model into the minio pod.
    :param model_type: The type of the model.
    :param model: Model.
    :param artifact_path: Relative path of the artifact in the minio pod.
    :param registered_model_name: If none None, register model into the model registry with
                                  this name. If None, the model only be saved in minio pod.
    """
    SubmarineClient().save_model(
        model_type, model, artifact_path, registered_model_name, input_dim, output_dim
    )
