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

import os

from submarine.client.api_client import ApiClient
from submarine.client.configuration import Configuration


def generate_host() -> str:
    """
    Generate submarine host
    :return: submarine host
    """
    submarine_server_dns_name = str(os.environ.get("SUBMARINE_SERVER_DNS_NAME"))
    submarine_server_port = str(os.environ.get("SUBMARINE_SERVER_PORT"))
    host = "http://" + submarine_server_dns_name + ":" + submarine_server_port
    return host


def get_api_client(host: str) -> ApiClient:
    configuration = Configuration()
    configuration.host = host + "/api"
    api_client = ApiClient(configuration=configuration)

    return api_client
