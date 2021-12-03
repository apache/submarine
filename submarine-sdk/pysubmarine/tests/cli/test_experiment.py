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


def test_list_experiment():
    runner = CliRunner()
    result = runner.invoke(main.entry_point, ["list", "experiment"])
    assert result.exit_code == 0
    assert "list experiment!" in result.output


def test_get_experiment():
    mock_experiment_id = "0"
    runner = CliRunner()
    result = runner.invoke(main.entry_point, ["get", "experiment", mock_experiment_id])
    assert result.exit_code == 0
    assert "get experiment! id={}".format(mock_experiment_id) in result.output


def test_delete_experiment():
    mock_experiment_id = "0"
    runner = CliRunner()
    result = runner.invoke(main.entry_point, ["delete", "experiment", mock_experiment_id])
    assert result.exit_code == 0
    assert "delete experiment! id={}".format(mock_experiment_id) in result.output
