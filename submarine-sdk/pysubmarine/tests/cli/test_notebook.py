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
from submarine.client.api.notebook_client import NotebookClient
from submarine.client.models.environment_spec import EnvironmentSpec
from submarine.client.models.notebook_meta import NotebookMeta
from submarine.client.models.notebook_pod_spec import NotebookPodSpec
from submarine.client.models.notebook_spec import NotebookSpec

TEST_CONSOLE_WIDTH = 191


@pytest.mark.e2e
def test_all_notbook_e2e():
    """E2E Test for using submarine CLI to access submarine notebook
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
    assert "connection.port=8080" in result.output

    submarine_client = NotebookClient(host="http://localhost:8080")

    mock_user_id = "4291d7da9005377ec9aec4a71ea837f"

    notebook_meta = NotebookMeta(name="test-nb", namespace="default", owner_id=mock_user_id)
    environment = EnvironmentSpec(name="notebook-env")
    notebook_podSpec = NotebookPodSpec(env_vars={"TEST_ENV": "test"}, resources="cpu=1,memory=1.0Gi")
    notebookSpec = NotebookSpec(meta=notebook_meta, environment=environment, spec=notebook_podSpec)

    notebook = submarine_client.create_notebook(notebookSpec)
    notebookId = notebook["notebookId"]

    # test list notebook
    result = runner.invoke(main.entry_point, ["list", "notebook"])
    assert result.exit_code == 0
    assert "List of Notebooks" in result.output
    assert notebook["name"] in result.output
    assert notebook["notebookId"] in result.output
    assert notebook["spec"]["environment"]["name"] in result.output
    assert notebook["spec"]["spec"]["resources"] in result.output
    # no need to check status (we do not wait for the notbook to run)

    # test get notebook
    result = runner.invoke(main.entry_point, ["get", "notebook", notebookId])
    assert f"Notebook(id = {notebookId} )" in result.output
    assert notebook["spec"]["environment"]["name"] in result.output

    # test delete notebook
    result = runner.invoke(main.entry_point, ["delete", "notebook", notebookId])
    assert f"Notebook(id = {notebookId} ) deleted" in result.output

    # test get environment fail after delete
    result = runner.invoke(main.entry_point, ["get", "notebook", notebookId])
    assert "[Api Error] Notebook not found." in result.output
