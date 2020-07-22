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
import time

from submarine.tracking.client import SubmarineClient
from submarine.tracking.utils import get_job_id

_RUN_ID_ENV_VAR = "SUBMARINE_RUN_ID"
_active_run_stack = []

_logger = logging.getLogger(__name__)


def log_param(key, value, worker_index):
    """
    Log a parameter under the current run, creating a run if necessary.
    :param key: Parameter name (string)
    :param value: Parameter value (string, but will be string-field if not)
    :param worker_index
    """
    job_id = get_job_id()
    SubmarineClient().log_param(job_id, key, value, worker_index)


def log_metric(key, value, worker_index, step=None):
    """
    Log a metric under the current run, creating a run if necessary.
    :param key: Metric name (string).
    :param value: Metric value (float). Note that some special values such as +/- Infinity may be
                  replaced by other values depending on the store. For example, sFor example, the
                  SQLAlchemy store replaces +/- Inf with max / min float values.
    :param worker_index: Metric worker_index (string).
    :param step: Metric step (int). Defaults to zero if unspecified.
    """
    job_name = get_job_id()
    SubmarineClient().log_metric(job_name, key, value, worker_index,
                                 int(time.time() * 1000), step or 0)
