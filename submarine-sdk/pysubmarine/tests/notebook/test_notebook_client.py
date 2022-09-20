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

from submarine.client.api.notebook_client import NotebookClient
from submarine.client.models.environment_spec import EnvironmentSpec
from submarine.client.models.notebook_meta import NotebookMeta
from submarine.client.models.notebook_pod_spec import NotebookPodSpec
from submarine.client.models.notebook_spec import NotebookSpec


@pytest.mark.e2e
def test_notebook_e2e():
    submarine_client = NotebookClient(host="http://localhost:8080")

    mock_user_id = "4291d7da9005377ec9aec4a71ea837f"

    notebook_meta = NotebookMeta(name="test-nb", namespace="default", owner_id=mock_user_id)
    environment = EnvironmentSpec(name="notebook-env")
    notebook_podSpec = NotebookPodSpec(env_vars={"TEST_ENV": "test"}, resources="cpu=1,memory=1.0Gi")
    notebookSpec = NotebookSpec(meta=notebook_meta, environment=environment, spec=notebook_podSpec)

    notebook = submarine_client.create_notebook(notebookSpec)

    notebookId = notebook["notebookId"]
    submarine_client.get_notebook(notebookId)
    submarine_client.list_notebooks(user_id=mock_user_id)
    submarine_client.delete_notebook(notebookId)
