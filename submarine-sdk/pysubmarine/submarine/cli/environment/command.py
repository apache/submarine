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

import json
import time

import click
from rich.console import Console
from rich.json import JSON as richJSON
from rich.panel import Panel
from rich.table import Table

from submarine.cli.config.config import loadConfig
from submarine.client.api.environment_client import EnvironmentClient
from submarine.client.exceptions import ApiException

submarineCliConfig = loadConfig()

environmentClient = EnvironmentClient(
    host=f"http://{submarineCliConfig.connection.hostname}:{submarineCliConfig.connection.port}"
)

POLLING_INTERVAL = 1  # sec
TIMEOUT = 30  # sec


@click.command("environment")
def list_environment():
    """List environment"""
    COLS_TO_SHOW = ["Name", "Id", "dockerImage"]
    console = Console()
    try:
        thread = environmentClient.list_environments_async()
        timeout = time.time() + TIMEOUT
        with console.status("[bold green] Fetching Environments..."):
            while not thread.ready():
                time.sleep(POLLING_INTERVAL)
                if time.time() > timeout:
                    console.print("[bold red] Timeout!")
                    return

        result = thread.get()
        results = result.result

        results = list(
            map(
                lambda r: [
                    r["environmentSpec"]["name"],
                    r["environmentId"],
                    r["environmentSpec"]["dockerImage"],
                ],
                results,
            )
        )

        table = Table(title="List of Environments")

        for col in COLS_TO_SHOW:
            table.add_column(col, overflow="fold")
        for res in results:
            table.add_row(*res)

        console.print(table)

    except ApiException as err:
        if err.body is not None:
            errbody = json.loads(err.body)
            click.echo("[Api Error] {}".format(errbody["message"]))
        else:
            click.echo(f"[Api Error] {err}")


@click.command("environment")
@click.argument("name")
def get_environment(name):
    """Get environment"""
    console = Console()
    try:
        thread = environmentClient.get_environment_async(name)
        timeout = time.time() + TIMEOUT
        with console.status(f"[bold green] Fetching Environment(name = {name} )..."):
            while not thread.ready():
                time.sleep(POLLING_INTERVAL)
                if time.time() > timeout:
                    console.print("[bold red] Timeout!")
                    return

        result = thread.get()
        result = result.result

        json_data = richJSON.from_data(result)
        console.print(Panel(json_data, title=f"Environment(name = {name} )"))
    except ApiException as err:
        if err.body is not None:
            errbody = json.loads(err.body)
            click.echo("[Api Error] {}".format(errbody["message"]))
        else:
            click.echo(f"[Api Error] {err}")


@click.command("environment")
@click.argument("name")
def delete_environment(name):
    """Delete environment"""
    console = Console()
    try:
        thread = environmentClient.delete_environment_async(name)
        timeout = time.time() + TIMEOUT
        with console.status(f"[bold green] Deleting Environment(name = {name} )..."):
            while not thread.ready():
                time.sleep(POLLING_INTERVAL)
                if time.time() > timeout:
                    console.print("[bold red] Timeout!")
                    return

        result = thread.get()
        result = result.result

        console.print(f"[bold green] Environment(name = {name} ) deleted")

    except ApiException as err:
        if err.body is not None:
            errbody = json.loads(err.body)
            click.echo("[Api Error] {}".format(errbody["message"]))
        else:
            click.echo(f"[Api Error] {err}")
