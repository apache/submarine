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
from click.testing import CliRunner

from submarine.cli import main
from submarine.client.api.environment_client import EnvironmentClient
from submarine.client.models.environment_spec import EnvironmentSpec
from submarine.client.models.kernel_spec import KernelSpec

TEST_CONSOLE_WIDTH = 191


@pytest.mark.e2e
def test_all_environment_e2e():
    """E2E Test for using submarine CLI to access submarine environment
    To run this test, you should first set
        your submarine CLI config `port` to 8080 and `hostname` to localhost
    i.e. please execute the commands in your terminal:
        submarine config set connection.hostname localhost
        submarine config set connection.port 8080
    """
    # set env to display full table
    runner = CliRunner(env={"COLUMNS": str(TEST_CONSOLE_WIDTH)})
    # check if cli config is correct for testing
    result = runner.invoke(main.entry_point, ["config", "get", "connection.port"])
    assert result.exit_code == 0
    assert f"connection.port={8080}" in result.output

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
        docker_image="apache/submarine:jupyter-notebook-gpu-0.8.0-SNAPSHOT",
    )

    environment = submarine_client.create_environment(environment_spec=environment_spec)
    environment_name = environment["environmentSpec"]["name"]

    # test list environment
    result = runner.invoke(main.entry_point, ["list", "environment"])
    assert result.exit_code == 0
    assert "List of Environments" in result.output
    assert environment["environmentSpec"]["name"] in result.output
    assert environment["environmentSpec"]["dockerImage"] in result.output
    assert environment["environmentId"] in result.output

    # test get environment
    result = runner.invoke(main.entry_point, ["get", "environment", environment_name])
    assert f"Environment(name = {environment_name} )" in result.output
    assert environment["environmentSpec"]["name"] in result.output

    # test delete environment
    result = runner.invoke(main.entry_point, ["delete", "environment", environment_name])
    assert f"Environment(name = {environment_name} ) deleted" in result.output

    # test get environment fail after delete
    result = runner.invoke(main.entry_point, ["get", "environment", environment_name])
    assert "[Api Error] Environment not found." in result.output
