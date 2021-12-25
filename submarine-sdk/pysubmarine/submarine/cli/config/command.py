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

from dataclasses import asdict
from typing import Union

import click
from rich.console import Console
from rich.json import JSON as richJSON
from rich.panel import Panel

from submarine.cli.config.config import initConfig, loadConfig, rgetattr, rsetattr, saveConfig


@click.command("list")
def list_config():
    """List Submarine CLI Config"""
    console = Console()
    _config = loadConfig()
    json_data = richJSON.from_data({**asdict(_config)})
    console.print(Panel(json_data, title="SubmarineCliConfig"))


@click.command("get")
@click.argument("param")
def get_config(param):
    """Get Submarine CLI Config"""
    _config = loadConfig()
    try:
        click.echo("{}={}".format(param, rgetattr(_config, param)))
    except AttributeError as err:
        click.echo(err)


@click.command("set")
@click.argument("param")
@click.argument("value")
def set_config(param, value):
    """Set Submarine CLI Config"""
    _config = loadConfig()
    _paramField = rgetattr(_config, param)
    # define types that can be cast from command line input
    primitive = (int, str, bool)

    def is_primitiveType(_type):
        return _type in primitive

    # cast type
    if type(_paramField) == type(Union) and is_primitiveType(type(_paramField).__args__[0]):
        value = type(_paramField).__args__[0](value)
    elif is_primitiveType(type(_paramField)):
        value = type(_paramField)(value)

    try:
        rsetattr(_config, param, value)
    except TypeError as err:
        click.echo(err)
    saveConfig(_config)


@click.command("init")
def init_config():
    """Init Submarine CLI Config"""
    try:
        initConfig()
        click.echo("Submarine CLI Config initialized")
    except AttributeError as err:
        click.echo(err)
