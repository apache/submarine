"""
 Licensed to the Apache Software Foundation (ASF) under one
 or more contributor license agreements.  See the NOTICE file
 distributed with this work for additional information
 regarding copyright ownership.  The ASF licenses this file
 to you under the Apache License, Version 2.0 (the
 "License"); you may not use this file except in compliance
 with the License.  You may obtain a copy of the License at
 http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing,
 software distributed under the License is distributed on an
 "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 KIND, either express or implied.  See the License for the
 specific language governing permissions and limitations
 under the License.
"""

import click

from submarine.cli.config import command as config_cmd
from submarine.cli.environment import command as environment_cmd
from submarine.cli.experiment import command as experiment_cmd
from submarine.cli.notebook import command as notebook_cmd
from submarine.cli.sandbox import command as sandbox_cmd


@click.group()
def entry_point():
    """Submarine CLI Tool!"""
    pass


@entry_point.group("list")
def cmdgrp_list():
    pass


@entry_point.group("get")
def cmdgrp_get():
    pass


@entry_point.group("delete")
def cmdgrp_delete():
    pass


@entry_point.group("sandbox")
def cmdgrp_sandbox():
    pass


@entry_point.group("config")
def cmdgrp_config():
    pass


# experiment
cmdgrp_list.add_command(experiment_cmd.list_experiment)
cmdgrp_get.add_command(experiment_cmd.get_experiment)
cmdgrp_delete.add_command(experiment_cmd.delete_experiment)
# notebook
cmdgrp_list.add_command(notebook_cmd.list_notebook)
cmdgrp_get.add_command(notebook_cmd.get_notebook)
cmdgrp_delete.add_command(notebook_cmd.delete_notebook)
# environment
cmdgrp_list.add_command(environment_cmd.list_environment)
cmdgrp_get.add_command(environment_cmd.get_environment)
cmdgrp_delete.add_command(environment_cmd.delete_environment)

# sandbox
cmdgrp_sandbox.add_command(sandbox_cmd.start_sandbox)
cmdgrp_sandbox.add_command(sandbox_cmd.delete_sandbox)

# config
cmdgrp_config.add_command(config_cmd.set_config)
cmdgrp_config.add_command(config_cmd.list_config)
cmdgrp_config.add_command(config_cmd.get_config)
cmdgrp_config.add_command(config_cmd.init_config)
