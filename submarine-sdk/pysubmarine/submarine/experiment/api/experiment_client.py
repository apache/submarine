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

import logging
import os
import time

from submarine.experiment.configuration import Configuration
from submarine.experiment.api_client import ApiClient
from submarine.experiment.api.experiment_api import ExperimentApi

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(message)s')
logging.getLogger().setLevel(logging.INFO)


def generate_host():
    """
    Generate submarine host
    :return: submarine host
    """
    submarine_server_dns_name = str(os.environ.get("SUBMARINE_SERVER_DNS_NAME"))
    submarine_server_port = str(os.environ.get("SUBMARINE_SERVER_PORT"))
    host = submarine_server_dns_name + ':' + submarine_server_port
    return host


class ExperimentClient:
    def __init__(self, host=generate_host()):
        """
        Submarine experiment client constructor
        :param host: An HTTP URI like http://submarine-server:8080.
        """
        # TODO(pingsutw): support authentication for talking to the submarine server
        self.host = host
        configuration = Configuration()
        configuration.host = host + '/api'
        api_client = ApiClient(configuration=configuration)
        self.experiment_api = ExperimentApi(api_client=api_client)

    def create_experiment(self, experiment_spec):
        """
        Create an experiment
        :param experiment_spec: submarine experiment spec
        :return: submarine experiment
        """
        response = self.experiment_api.create_experiment(experiment_spec=experiment_spec)
        return response.result

    def wait_for_finish(self, id, polling_interval=10):
        """
        Waits until experiment is finished or failed
        :param id: submarine experiment id
        :param polling_interval: How often to poll for the status of the experiment.
        :return: str: experiment logs
        """
        index = 0
        while True:
            status = self.get_experiment(id)['status']
            if status == 'Succeeded' or status == 'Deleted':
                self._log_pod(id, index)
                break
            index = self._log_pod(id, index)
            time.sleep(polling_interval)

    def _log_pod(self, id, index):
        response = self.experiment_api.get_log(id)
        log_contents = response.result['logContent']
        if len(log_contents) == 0:
            return index
        log_content = log_contents[0]
        for i, log in enumerate(log_content['podLog']):
            if i < index:
                continue
            index += 1
            logging.info("%s", log)
        return index

    def patch_experiment(self, id, experiment_spec):
        """
        Patch an experiment
        :param id: submarine experiment id
        :param experiment_spec: submarine experiment spec
        :return: submarine experiment
        """
        response = self.experiment_api.patch_experiment(id=id, experiment_spec=experiment_spec)
        return response.result

    def get_experiment(self, id):
        """
        Get the experiment's detailed info by id
        :param id: submarine experiment id
        :return: submarine experiment
        """
        response = self.experiment_api.get_experiment(id=id)
        return response.result

    def list_experiments(self, status=None):
        """
        List all experiment for the user
        :param status: Accepted, Created, Running, Succeeded, Deleted
        :return: List of submarine experiments
        """
        response = self.experiment_api.list_experiments(status=status)
        return response.result

    def delete_experiment(self, id):
        """
        Delete the Submarine experiment
        :param id: Submarine experiment id
        :return: The detailed info about deleted submarine experiment
        """
        response = self.experiment_api.delete_experiment(id)
        return response.result

    def get_log(self, id, onlyMaster=False):
        """
        Get training logs of all pod of the experiment.
        By default get all the logs of Pod
        :param id: experiment id
        :param onlyMaster: By default include pod log of "master" which might be
         Tensorflow PS/Chief or PyTorch master
        :return: str: pods logs
        """
        response = self.experiment_api.get_log(id)
        log_contents = response.result['logContent']

        if onlyMaster is True and len(log_contents) != 0:
            log_contents = [log_contents[0]]

        for log_content in log_contents:
            logging.info("The logs of Pod %s:\n", log_content['podName'])
            for log in log_content['podLog']:
                logging.info("%s", log)

    def list_log(self, status):
        """
        List experiment log
        :param status: Accepted, Created, Running, Succeeded, Deleted
        :return: List of submarine log
        """
        response = self.experiment_api.list_log(status=status)
        return response.result
