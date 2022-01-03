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

from submarine.client.api.serve_api import ServeApi
from submarine.client.utils.api_utils import generate_host, get_api_client

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(message)s")
logging.getLogger().setLevel(logging.INFO)


class ServeClient:
    def __init__(self, host: str = generate_host()) -> None:
        """
        Submarine serve client constructor
        :param host: An HTTP URI like http://submarine-server:8080.
        """
        api_client = get_api_client(host)
        self.serve_api = ServeApi(api_client=api_client)

    def create_serve(self, model_name: str, model_version: int):
        """
        Create a model serve
        :param model_name: Name of a registered model
        :param model_version: Version of a registered model
        """
        spec = {"modelName": model_name, "modelVersion": model_version}
        response = self.serve_api.create_serve(*spec)
        return response.result

    def delete_serve(self, model_name: str, model_version: int):
        """
        Delete a serving model
        :param model_name: Name of a registered model
        :param model_version: Version of a registered model
        """
        spec = {"modelName": model_name, "modelVersion": model_version}
        response = self.serve_api.create_serve(*spec)
        return response.result
