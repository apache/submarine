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
import time

from submarine.entities import Metric, Param
from submarine.tracking import utils
from submarine.utils.validation import validate_metric, validate_param


class SubmarineClient(object):
    """
    Client of an submarine Tracking Server that creates and manages experiments and runs.
    """

    def __init__(self, tracking_uri=None):
        """
        :param tracking_uri: Address of local or remote tracking server. If not provided, defaults
                             to the service set by ``submarine.tracking.set_tracking_uri``. See
                             `Where Runs Get Recorded <../tracking.html#where-runs-get-recorded>`_
                             for more info.
        """
        self.tracking_uri = tracking_uri or utils.get_tracking_uri()
        self.store = utils.get_sqlalchemy_store(self.tracking_uri)

    def log_metric(self,
                   job_id,
                   key,
                   value,
                   worker_index,
                   timestamp=None,
                   step=None):
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
        timestamp = timestamp if timestamp is not None else int(time.time())
        step = step if step is not None else 0
        validate_metric(key, value, timestamp, step)
        metric = Metric(key, value, worker_index, timestamp, step)
        self.store.log_metric(job_id, metric)

    def log_param(self, job_id, key, value, worker_index):
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
