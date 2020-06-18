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
from submarine.experiment.configuration import Configuration
from submarine.experiment.api_client import ApiClient
from submarine.experiment.api.experiment_api import ExperimentApi

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(message)s')
logging.getLogger().setLevel(logging.INFO)


class ExperimentClient:
    def __init__(self, host):
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

    def wait_for_finish(self, id, timeout_seconds=600, polling_interval=30):
        """
        Waits until experiment is finished or failed
        :param id: submarine experiment id
        :param timeout_seconds: How long to wait for the experiment. Default is 600s
        :param polling_interval: How often to poll for the status of the experiment.
        :return: str: experiment logs
        """
        # TODO(pingsutw): Support continue log experiment until experiment is finished or failed
        raise NotImplementedError("To be implemented")

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

    def list_experiments(self, status):
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

    def get_log(self, id, master=True):
        """
        Get training logs of the experiment.
        By default only get the logs of Pod that has labels 'job-role: master'.
        :param master: By default get pod with label 'job-role: master' pod if True.
                    If need to get more Pod Logs, set False.
        :param id: Experiment ID
        :return: str: experiment logs
        """
        response = self.experiment_api.get_log(id)
        log_contents = response.result['logContent']

        if master is True:
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
