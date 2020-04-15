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

from submarine.utils.rest_utils import http_request
import logging
import json

_logger = logging.getLogger(__name__)

_PATH_PREFIX = "/api/v1/"
JOBS = 'jobs'


class SubmarineJobClient:
    def __init__(self, hostname, port):
        self.base_url = 'http://' + hostname + ':' + str(port)

    def submit_job(self, conf_path):
        """
        Submit a job to submarine server
        :param conf_path: The location of the configuration file
        :return: requests.Response
        """
        endpoint = _PATH_PREFIX + JOBS
        with open(conf_path) as json_file:
            json_body = json.load(json_file)
        response = http_request(self.base_url, endpoint=endpoint,
                                method='POST', json_body=json_body)
        return response

    def delete_job(self, job_id):
        """
        delete a submarine job
        :param job_id: submarine job ID
        :return: requests.Response: the detailed info about deleted job
        """
        endpoint = _PATH_PREFIX + JOBS + '/' + job_id
        response = http_request(self.base_url, endpoint=endpoint,
                                method='DELETE', json_body=None)
        return response
