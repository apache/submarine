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

import pytest

from submarine.client.api.environment_client import EnvironmentClient
from submarine.client.models.environment_spec import EnvironmentSpec
from submarine.client.models.kernel_spec import KernelSpec


@pytest.mark.e2e
def test_environment_e2e():
    submarine_client = EnvironmentClient(host="http://localhost:8080")
    kernel_spec = KernelSpec(
        name="submarine_jupyter_py3",
        channels=["defaults"],
        conda_dependencies=[],
        pip_dependencies=[],
    )
    environment_spec = EnvironmentSpec(
        name="mytest",
        kernel_spec=kernel_spec,
        docker_image="apache/submarine:jupyter-notebook-gpu-0.8.0-RC0",
    )

    environment = submarine_client.create_environment(environment_spec=environment_spec)
    environment_name = environment["environmentSpec"]["name"]
    submarine_client.get_environment(environment_name)
    submarine_client.list_environments()
    submarine_client.delete_environment(environment_name)
