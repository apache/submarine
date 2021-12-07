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

from click.testing import CliRunner

from submarine.cli import main


def test_list_notebook():
    runner = CliRunner()
    result = runner.invoke(main.entry_point, ["list", "notebook"])
    assert result.exit_code == 0
    assert "list notebook!" in result.output


def test_get_notebook():
    mock_notebook_id = "0"
    runner = CliRunner()
    result = runner.invoke(main.entry_point, ["get", "notebook", mock_notebook_id])
    assert result.exit_code == 0
    assert "get notebook! id={}".format(mock_notebook_id) in result.output


def test_delete_notebook():
    mock_notebook_id = "0"
    runner = CliRunner()
    result = runner.invoke(main.entry_point, ["delete", "notebook", mock_notebook_id])
    assert result.exit_code == 0
    assert "delete notebook! id={}".format(mock_notebook_id) in result.output
