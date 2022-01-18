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

from submarine.client.api.notebook_api import NotebookApi
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
    host = "http://" + submarine_server_dns_name + ":" + submarine_server_port
    return host


class NotebookClient:
    def __init__(self, host: str = generate_host()):
        """
        Submarine notebook client constructor
        :param host: An HTTP URI like http://submarine-server:8080.
        """
        # TODO(pingsutw): support authentication for talking to the submarine server
        self.host = host
        configuration = Configuration()
        configuration.host = host + "/api"
        api_client = ApiClient(configuration=configuration)
        self.notebook_api = NotebookApi(api_client=api_client)

    def create_notebook(self, notebook_spec):
        """
        Create an notebook
        :param notebook_spec: submarine notebook spec
        :return: submarine notebook
        """
        response = self.notebook_api.create_notebook(notebook_spec=notebook_spec)
        return response.result

    def get_notebook(self, id):
        """
        Get the notebook's detailed info by id
        :param id: submarine notebook id
        :return: submarine notebook
        """
        response = self.notebook_api.get_notebook(id=id)
        return response.result

    def get_notebook_async(self, id):
        """
        Get the notebook's detailed info by id (async)
        :param id: submarine notebook id
        :return: multiprocessing.pool.ApplyResult
        """
        thread = self.notebook_api.get_notebook(id=id, async_req=True)
        return thread

    def list_notebooks(self, user_id):
        """
        List notebook instances which belong to user
        :param user_id
        :return: List of submarine notebooks
        """
        response = self.notebook_api.list_notebooks(id=user_id)
        return response.result

    def list_notebooks_async(self, user_id):
        """
        List notebook instances which belong to user (async)
        :param user_id:
        :return: multiprocessing.pool.ApplyResult
        """
        thread = self.notebook_api.list_notebooks(id=user_id, async_req=True)
        return thread

    def delete_notebook(self, id):
        """
        Delete the Submarine notebook
        :param id: Submarine notebook id
        :return: The detailed info about deleted submarine notebook
        """
        response = self.notebook_api.delete_notebook(id)
        return response.result

    def delete_notebook_async(self, id):
        """
        Delete the Submarine notebook (async)
        :param id: Submarine notebook id
        :return: The detailed info about deleted submarine notebook
        """
        thread = self.notebook_api.delete_notebook(id, async_req=True)
        return thread
