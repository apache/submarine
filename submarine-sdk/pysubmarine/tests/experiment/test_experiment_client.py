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

import submarine
from submarine.experiment.models.code_spec import CodeSpec
from submarine.experiment.models.environment_spec import EnvironmentSpec
from submarine.experiment.models.experiment_meta import ExperimentMeta
from submarine.experiment.models.experiment_spec import ExperimentSpec
from submarine.experiment.models.experiment_task_spec import ExperimentTaskSpec


@pytest.mark.e2e
def test_experiment_e2e():
    submarine_client = submarine.ExperimentClient(host='http://localhost:8080')
    environment = EnvironmentSpec(
        image='gcr.io/kubeflow-ci/tf-dist-mnist-test:1.0')
    experiment_meta = ExperimentMeta(
        name='mnist-dist',
        namespace='default',
        framework='Tensorflow',
        cmd='python /var/tf_dist_mnist/dist_mnist.py --train_steps=100',
        env_vars={'ENV1': 'ENV1'})

    worker_spec = ExperimentTaskSpec(resources='cpu=1,memory=1024M', replicas=1)
    ps_spec = ExperimentTaskSpec(resources='cpu=1,memory=1024M', replicas=1)

    code_spec = CodeSpec(sync_mode='git',
                         url='https://github.com/apache/submarine.git')

    experiment_spec = ExperimentSpec(meta=experiment_meta,
                                     environment=environment,
                                     code=code_spec,
                                     spec={
                                         'Ps': ps_spec,
                                         'Worker': worker_spec
                                     })

    experiment = submarine_client.create_experiment(
        experiment_spec=experiment_spec)
    id = experiment['experimentId']

    submarine_client.get_experiment(id)
    submarine_client.list_experiments()
    submarine_client.get_log(id)
    submarine_client.delete_experiment(id)
