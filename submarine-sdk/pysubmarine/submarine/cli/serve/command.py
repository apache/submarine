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

from submarine.cli.config.config import loadConfig
from submarine.client.api.serve_client import ServeClient
from submarine.client.api_client import ApiException

submarineCliConfig = loadConfig()
serveClient = ServeClient(
    f"http://{submarineCliConfig.connection.hostname}:{submarineCliConfig.connection.port}"
    if submarineCliConfig
    else "http://localhost:32080"
)

POLLING_INTERVAL = 1  # sec
TIMEOUT = 30  # sec


@click.command("serve")
def list_serve():
    """List serves"""
    click.echo("The command is not supported yet.")  # TODO(kuanhsun)
    click.echo("list serve!")


@click.command("serve")
@click.argument("model_name")
@click.argument("model_version", type=int)
def get_serve(model_name: str, model_version: int):
    """Get serve"""
    click.echo("The command is not supported yet.")  # TODO(kuanhsun)
    click.echo(f"get serve! model name: {model_name}, model version: {model_version}")


@click.command("serve")
@click.argument("model_name")
@click.argument("model_version", type=int)
def create_serve(model_name: str, model_version: int):
    """Create serve"""
    console = Console()
    try:
        thread = serveClient.create_serve(model_name, model_version, async_req=True)
        timeout = time.time() + TIMEOUT
        with console.status(
            f"[bold green] Creating Serve with name: {model_name}, version: {model_version}"
        ):
            while not thread.ready():
                time.sleep(POLLING_INTERVAL)
                if time.time() > timeout:
                    console.print("[bold red] Timeout!")
                    return
        result = thread.get()
        click.echo(result)

    except ApiException as err:
        if err.body is not None:
            errbody = json.loads(err.body)
            click.echo(f"[Api Error] {errbody['message']}")
        else:
            click.echo(f"[Api Error] {err}")


@click.command("serve")
@click.argument("model_name")
@click.argument("model_version", type=int)
@click.option("--wait", is_flag=True, default=False)
def delete_serve(model_name: str, model_version: int, wait: bool):
    """Delete serve"""
    console = Console()
    try:
        thread = serveClient.delete_serve(model_name, model_version, async_req=True)
        timeout = time.time() + TIMEOUT
        with console.status(
            f"[bold green] Deleting Serve with name: {model_name}, version: {model_version}"
        ):
            while not thread.ready():
                time.sleep(POLLING_INTERVAL)
                if time.time() > timeout:
                    console.print("[bold red] Timeout!")
                    return

        result = thread.get()
        click.echo(result)

        if wait:
            if result["status"] == "Deleted":
                console.print(
                    f"[bold green] Serve: model name:{model_name}, version: {model_version}"
                )
            else:
                console.print("[bold red] Failed")
                json_data = richJSON.from_data(result)
                console.print(
                    Panel(
                        json_data, title=f"Serve: model name:{model_name}, version: {model_version}"
                    )
                )
    except ApiException as err:
        if err.body is not None:
            errbody = json.loads(err.body)
            click.echo(f"[Api Error] {errbody['message']}")
        else:
            click.echo(f"[Api Error] {err}")
