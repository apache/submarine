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

import submarine
from submarine.cli import main
from submarine.client.models.code_spec import CodeSpec
from submarine.client.models.environment_spec import EnvironmentSpec
from submarine.client.models.experiment_meta import ExperimentMeta
from submarine.client.models.experiment_spec import ExperimentSpec
from submarine.client.models.experiment_task_spec import ExperimentTaskSpec

TEST_CONSOLE_WIDTH = 191


@pytest.mark.e2e
def test_all_experiment_e2e():
    """E2E Test for using submarine CLI to access submarine experiment
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

    submarine_client = submarine.ExperimentClient(host="http://localhost:8080")
    environment = EnvironmentSpec(image="apache/submarine:tf-dist-mnist-test-1.0")
    experiment_meta = ExperimentMeta(
        name="mnist-dist",
        namespace="default",
        framework="Tensorflow",
        cmd="python /var/tf_dist_mnist/dist_mnist.py --train_steps=100",
        env_vars={"ENV1": "ENV1"},
    )

    worker_spec = ExperimentTaskSpec(resources="cpu=1,memory=1024M", replicas=1)
    ps_spec = ExperimentTaskSpec(resources="cpu=1,memory=1024M", replicas=1)

    code_spec = CodeSpec(sync_mode="git", url="https://github.com/apache/submarine.git")

    experiment_spec = ExperimentSpec(
        meta=experiment_meta,
        environment=environment,
        code=code_spec,
        spec={"Ps": ps_spec, "Worker": worker_spec},
    )

    experiment = submarine_client.create_experiment(experiment_spec=experiment_spec)
    experiment = submarine_client.get_experiment(experiment["experimentId"])

    # test list experiment
    result = runner.invoke(main.entry_point, ["list", "experiment"])
    assert result.exit_code == 0
    assert "List of Experiments" in result.output
    assert experiment["spec"]["meta"]["name"] in result.output
    assert experiment["experimentId"] in result.output
    assert experiment["createdTime"] in result.output
    if experiment["runningTime"] is not None:
        assert experiment["runningTime"] in result.output
    if experiment["status"] is not None:
        assert experiment["status"] in result.output

    # test get experiment
    result = runner.invoke(main.entry_point, ["get", "experiment", experiment["experimentId"]])
    assert "Experiment(id = {} )".format(experiment["experimentId"]) in result.output
    assert experiment["spec"]["environment"]["image"] in result.output

    # test delete experiment (blocking mode)
    result = runner.invoke(
        main.entry_point, ["delete", "experiment", experiment["experimentId"], "--wait"]
    )
    assert "Experiment(id = {} ) deleted".format(experiment["experimentId"]) in result.output

    # test get experiment fail after delete
    result = runner.invoke(main.entry_point, ["get", "experiment", experiment["experimentId"]])
    assert "[Api Error] Not found experiment." in result.output
