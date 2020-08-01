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

from __future__ import print_function

import json
import os
import uuid

from submarine.store import DEFAULT_SUBMARINE_JDBC_URL
from submarine.store.sqlalchemy_store import SqlAlchemyStore
from submarine.utils import env

_TRACKING_URI_ENV_VAR = "SUBMARINE_TRACKING_URI"
# https://github.com/linkedin/TonY/pull/431
_JOB_ID_ENV_VAR = "JOB_ID"

_TF_CONFIG = "TF_CONFIG"
_CLUSTER_SPEC = "CLUSTER_SPEC"
_JOB_NAME = "JOB_NAME"
_TYPE = "type"
_TASK = "task"
_INDEX = "index"
_RANK = "RANK"

# Extra environment variables which take precedence for setting the basic/bearer
# auth on http requests.
_TRACKING_USERNAME_ENV_VAR = "SUBMARINE_TRACKING_USERNAME"
_TRACKING_PASSWORD_ENV_VAR = "SUBMARINE_TRACKING_PASSWORD"
_TRACKING_TOKEN_ENV_VAR = "SUBMARINE_TRACKING_TOKEN"
_TRACKING_INSECURE_TLS_ENV_VAR = "SUBMARINE_TRACKING_INSECURE_TLS"

_tracking_uri = None


def is_tracking_uri_set():
    """Returns True if the tracking URI has been set, False otherwise."""
    if _tracking_uri or env.get_env(_TRACKING_URI_ENV_VAR):
        return True
    return False


def set_tracking_uri(uri):
    """
    Set the tracking server URI. This does not affect the
    currently active run (if one exists), but takes effect for successive runs.
    """
    global _tracking_uri
    _tracking_uri = uri


def get_tracking_uri():
    """
    Get the current tracking URI. This may not correspond to the tracking URI of
    the currently active run, since the tracking URI can be updated via ``set_tracking_uri``.
    :return: The tracking URI.
    """
    global _tracking_uri
    if _tracking_uri is not None:
        return _tracking_uri
    elif env.get_env(_TRACKING_URI_ENV_VAR) is not None:
        return env.get_env(_TRACKING_URI_ENV_VAR)
    else:
        return DEFAULT_SUBMARINE_JDBC_URL


def get_job_id():
    """
    Get the current experiment id.
    :return The experiment id:
    """
    # Get yarn application or K8s experiment ID when running distributed training
    if env.get_env(_JOB_ID_ENV_VAR) is not None:
        return env.get_env(_JOB_ID_ENV_VAR)
    else:  # set Random ID when running local training
        job_id = uuid.uuid4().hex
        os.environ[_JOB_ID_ENV_VAR] = job_id
        return job_id


def get_worker_index():
    """
    Get the current worker index.
    :return: The worker index:
    """
    # Get TensorFlow worker index
    if env.get_env(_TF_CONFIG) is not None:
        tf_config = json.loads(os.environ.get(_TF_CONFIG))
        task_config = tf_config.get(_TASK)
        task_type = task_config.get(_TYPE)
        task_index = task_config.get(_INDEX)
        worker_index = task_type + '-' + str(task_index)
    elif env.get_env(_CLUSTER_SPEC) is not None:
        cluster_spec = json.loads(os.environ.get(_CLUSTER_SPEC))
        task_config = cluster_spec.get(_TASK)
        task_type = task_config.get(_JOB_NAME)
        task_index = task_config.get(_INDEX)
        worker_index = task_type + '-' + str(task_index)
    # Get PyTorch worker index
    elif env.get_env(_RANK) is not None:
        rank = env.get_env(_RANK)
        if rank == "0":
            worker_index = "master-0"
        else:
            worker_index = "worker-" + rank
    # Set worker index to "worker-0" When running local training
    else:
        worker_index = "worker-0"

    return worker_index


def get_sqlalchemy_store(store_uri):
    return SqlAlchemyStore(store_uri)
