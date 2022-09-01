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

from submarine.client.api.environment_api import EnvironmentApi
from submarine.client.api_client import ApiClient
from submarine.client.configuration import Configuration

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(message)s")
logging.getLogger().setLevel(logging.INFO)


def generate_host():
    """
    Generate submarine host
    :return: submarine host
    """
    submarine_server_dns_name = str(os.environ.get("SUBMARINE_SERVER_DNS_NAME"))
    submarine_server_port = str(os.environ.get("SUBMARINE_SERVER_PORT"))
    host = submarine_server_dns_name + ":" + submarine_server_port
    return host


class EnvironmentClient:
    def __init__(self, host: str = generate_host()):
        """
        Submarine environment client constructor
        :param host: An HTTP URI like http://submarine-server:8080.
        """
        # TODO(pingsutw): support authentication for talking to the submarine server
        self.host = host
        configuration = Configuration()
        configuration.host = host + "/api"
        api_client = ApiClient(configuration=configuration)
        self.environment_api = EnvironmentApi(api_client=api_client)

    def create_environment(self, environment_spec):
        """
        Create an environment
        :param environment_spec: submarine environment spec
        :return: submarine environment
        """
        response = self.environment_api.create_environment(environment_spec=environment_spec)
        return response.result

    def create_environment_async(self, environment_spec):
        """
        Create an environment (async)
        :param environment_spec: submarine environment spec
        :return: thread
        """
        thread = self.environment_api.create_environment(
            environment_spec=environment_spec, async_req=True
        )
        return thread

    def update_environment(self, name, environment_spec):
        """
        Update an environment
        :param name: submarine environment name
        :param environment_spec: submarine environment spec
        :return: submarine environment
        """
        response = self.environment_api.update_environment(
            id=name, environment_spec=environment_spec
        )
        return response.result

    def get_environment(self, name):
        """
        Get the environment's detailed info by name
        :param name: submarine environment name
        :return: submarine environment
        """
        response = self.environment_api.get_environment(id=name)
        return response.result

    def get_environment_async(self, name):
        """
        Get the environment's detailed info by name (async)
        :param name: submarine environment name
        :return: thread
        """
        thread = self.environment_api.get_environment(id=name, async_req=True)
        return thread

    def list_environments(self, status=None):
        """
        List all environments for the user
        :param status:
        :return: List of submarine environments
        """
        response = self.environment_api.list_environment(status=status)
        return response.result

    def list_environments_async(self, status=None):
        """
        List all environments for the user (async)
        :param status:
        :return: thread
        """
        thread = self.environment_api.list_environment(status=status, async_req=True)
        return thread

    def delete_environment(self, name):
        """
        Delete the Submarine environment
        :param name: Submarine environment name
        :return: The detailed info about deleted submarine environment
        """
        response = self.environment_api.delete_environment(name)
        return response.result

    def delete_environment_async(self, name):
        """
        Delete the Submarine environment (async)
        :param name: Submarine environment name
        :return: thread
        """
        thread = self.environment_api.delete_environment(name, async_req=True)
        return thread
