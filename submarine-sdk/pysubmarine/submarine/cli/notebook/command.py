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
from submarine.client.api.notebook_client import NotebookClient
from submarine.client.exceptions import ApiException

submarineCliConfig = loadConfig()

notebookClient = NotebookClient(
    host=f"http://{submarineCliConfig.connection.hostname}:{submarineCliConfig.connection.port}"
)

POLLING_INTERVAL = 1  # sec
TIMEOUT = 30  # sec


@click.command("notebook")
def list_notebook():
    """List notebooks"""
    COLS_TO_SHOW = ["Name", "ID", "Environment", "Resources", "Status"]
    console = Console()
    # using user_id hard coded in SysUserRestApi.java
    # https://github.com/apache/submarine/blob/5040068d7214a46c52ba87e10e9fa64411293cf7/submarine-server/server-core/src/main/java/org/apache/submarine/server/workbench/rest/SysUserRestApi.java#L228
    try:
        thread = notebookClient.list_notebooks_async(user_id="4291d7da9005377ec9aec4a71ea837f")
        timeout = time.time() + TIMEOUT
        with console.status("[bold green] Fetching Notebook..."):
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
                    r["name"],
                    r["notebookId"],
                    r["spec"]["environment"]["name"],
                    r["spec"]["spec"]["resources"],
                    r["status"],
                ],
                results,
            )
        )

        table = Table(title="List of Notebooks")

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


@click.command("notebook")
@click.argument("id")
def get_notebook(id):
    """Get notebooks"""
    console = Console()
    try:
        thread = notebookClient.get_notebook_async(id)
        timeout = time.time() + TIMEOUT
        with console.status(f"[bold green] Fetching Notebook(id = {id} )..."):
            while not thread.ready():
                time.sleep(POLLING_INTERVAL)
                if time.time() > timeout:
                    console.print("[bold red] Timeout!")
                    return

        result = thread.get()
        result = result.result

        json_data = richJSON.from_data(result)
        console.print(Panel(json_data, title=f"Notebook(id = {id} )"))
    except ApiException as err:
        if err.body is not None:
            errbody = json.loads(err.body)
            click.echo("[Api Error] {}".format(errbody["message"]))
        else:
            click.echo(f"[Api Error] {err}")


@click.command("notebook")
@click.argument("id")
def delete_notebook(id):
    """Delete notebook"""
    console = Console()
    try:
        thread = notebookClient.delete_notebook_async(id)
        timeout = time.time() + TIMEOUT
        with console.status(f"[bold green] Deleting Notebook(id = {id} )..."):
            while not thread.ready():
                time.sleep(POLLING_INTERVAL)
                if time.time() > timeout:
                    console.print("[bold red] Timeout!")
                    return

        result = thread.get()
        result = result.result

        console.print(f"[bold green] Notebook(id = {id} ) deleted")

    except ApiException as err:
        if err.body is not None:
            errbody = json.loads(err.body)
            click.echo("[Api Error] {}".format(errbody["message"]))
        else:
            click.echo(f"[Api Error] {err}")
