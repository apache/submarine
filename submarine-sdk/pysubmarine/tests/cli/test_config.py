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
from submarine.cli.config.config import SubmarineCliConfig, initConfig, loadConfig


def test_list_config():
    initConfig()
    runner = CliRunner()
    result = runner.invoke(main.entry_point, ["config", "list"])
    _config = loadConfig()
    assert result.exit_code == 0
    assert "SubmarineCliConfig" in result.output
    assert '"hostname": "{}"'.format(_config.connection.hostname) in result.output
    assert '"port": {}'.format(_config.connection.port) in result.output


def test_init_config():
    runner = CliRunner()
    result = runner.invoke(main.entry_point, ["config", "init"])
    result = runner.invoke(main.entry_point, ["config", "list"])
    _default_config = SubmarineCliConfig()
    assert result.exit_code == 0
    assert '"hostname": "{}"'.format(_default_config.connection.hostname) in result.output
    assert '"port": {}'.format(_default_config.connection.port) in result.output


def test_get_set_experiment():
    initConfig()
    mock_hostname = "mockhost"
    runner = CliRunner()
    result = runner.invoke(main.entry_point, ["config", "get", "connection.hostname"])
    assert result.exit_code == 0
    _config = loadConfig()
    assert "connection.hostname={}".format(_config.connection.hostname) in result.output

    result = runner.invoke(
        main.entry_point, ["config", "set", "connection.hostname", mock_hostname]
    )
    assert result.exit_code == 0

    result = runner.invoke(main.entry_point, ["config", "get", "connection.hostname"])
    assert result.exit_code == 0
    _config = loadConfig()
    assert "connection.hostname={}".format(mock_hostname) in result.output
    assert mock_hostname == _config.connection.hostname
    initConfig()
