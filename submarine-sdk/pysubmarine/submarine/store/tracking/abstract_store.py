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

from abc import ABCMeta


class AbstractStore:
    """
    Abstract class for Backend Storage.
    This class defines the API interface for front ends to connect with various types of backends.
    """

    __metaclass__ = ABCMeta

    def __init__(self):
        """
        Empty constructor for now. This is deliberately not marked as abstract, else every
        derived class would be forced to create one.
        """
        pass

    def log_metric(self, job_id, metric):
        """
        Log a metric for the specified run
        :param job_id: String id for the run
        :param metric: :py:class:`submarine.entities.Metric` instance to log
        """
        pass

    def log_param(self, job_id, param):
        """
        Log a param for the specified run
        :param job_id: String id for the run
        :param param: :py:class:`submarine.entities.Param` instance to log
        """
        pass
