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

import functools
import os
from dataclasses import asdict, dataclass, field
from typing import Optional, Union

import click
import dacite
import yaml

CONFIG_YAML_PATH = os.path.join(os.path.dirname(__file__), "cli_config.yaml")


@dataclass
class BaseConfig:
    def __setattr__(self, __name, __value) -> None:
        """
        Override __setattr__ for custom type checking
        """
        # ignore this line for mypy checking, since there is some errors for mypy to check dataclass
        _field = self.__dataclass_fields__[__name]  # type: ignore
        if hasattr(_field.type, "__origin__") and _field.type.__origin__ == Union:
            if not isinstance(__value, _field.type.__args__):
                msg = (
                    "Field `{0.name}` is of type {1}, should be one of the type: {0.type.__args__}"
                    .format(_field, type(__value))
                )
                raise TypeError(msg)
        else:
            if not type(__value) == _field.type:
                msg = "Field {0.name} is of type {1}, should be {0.type}".format(
                    _field, type(__value)
                )
                raise TypeError(msg)

        super().__setattr__(__name, __value)


@dataclass
class ConnectionConfig(BaseConfig):
    hostname: Optional[str] = field(
        default="localhost",
        metadata={"help": "Hostname for submarine CLI to connect"},
    )

    port: Optional[int] = field(
        default=32080,
        metadata={"help": "Port for submarine CLI to connect"},
    )


@dataclass
class SubmarineCliConfig(BaseConfig):
    connection: ConnectionConfig = field(
        default_factory=lambda: ConnectionConfig(),
        metadata={"help": "Port for submarine CLI to connect"},
    )


def rgetattr(obj, attr, *args):
    """
    Recursive get attr
    Example:
        rgetattr(obj,"a.b.c") is equivalent to obj.a.b.c
    """

    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


def rsetattr(obj, attr, val):
    """
    Recursive set attr
    Example:
        rsetattr(obj,"a.b.c",val) is equivalent to obj.a.b.c = val
    """
    pre, _, post = attr.rpartition(".")
    if pre:
        _r = rgetattr(obj, pre)
        return setattr(_r, post, val)
    else:
        return setattr(obj, post, val)


def loadConfig(config_path: str = CONFIG_YAML_PATH) -> Optional[SubmarineCliConfig]:
    with open(config_path, "r") as stream:
        try:
            parsed_yaml: dict = yaml.safe_load(stream)
            return_config: SubmarineCliConfig = dacite.from_dict(
                data_class=SubmarineCliConfig, data=parsed_yaml
            )
            return return_config
        except yaml.YAMLError as exc:
            click.echo("Error Reading Config")
            click.echo(exc)
            return None


def saveConfig(config: SubmarineCliConfig, config_path: str = CONFIG_YAML_PATH):
    with open(config_path, "w") as stream:
        try:
            yaml.safe_dump({**asdict(config)}, stream)
        except yaml.YAMLError as exc:
            click.echo("Error Saving Config")
            click.echo(exc)


def initConfig(config_path: str = CONFIG_YAML_PATH):
    saveConfig(SubmarineCliConfig(), config_path)
