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

from submarine.experiment.api.experiment_client import ExperimentClient
from submarine.experiment.exceptions import ApiException

experimentClient = ExperimentClient("http://localhost:8080")

POLLING_INTERVAL = 1  # sec
TIMEOUT = 30  # sec


@click.command("experiment")
def list_experiment():
    """List experiments"""
    COLS_TO_SHOW = ["Name", "Id", "Tags", "Finished Time", "Created Time", "Running Time", "Status"]
    console = Console()
    try:
        thread = experimentClient.list_experiments_async()
        timeout = time.time() + TIMEOUT
        with console.status("[bold green] Fetching Experiments..."):
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
                    r["spec"]["meta"]["name"],
                    r["experimentId"],
                    ",".join(r["spec"]["meta"]["tags"]),
                    r["finishedTime"],
                    r["createdTime"],
                    r["runningTime"],
                    r["status"],
                ],
                results,
            )
        )

        table = Table(title="List of Experiments")

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
            click.echo("[Api Error] {}".format(err))


@click.command("experiment")
@click.argument("id")
def get_experiment(id):
    """Get experiments"""
    console = Console()
    try:
        thread = experimentClient.get_experiment_async(id)
        timeout = time.time() + TIMEOUT
        with console.status("[bold green] Fetching Experiment(id = {} )...".format(id)):
            while not thread.ready():
                time.sleep(POLLING_INTERVAL)
                if time.time() > timeout:
                    console.print("[bold red] Timeout!")
                    return

        result = thread.get()
        result = result.result

        json_data = richJSON.from_data(result)
        console.print(Panel(json_data, title="Experiment(id = {} )".format(id)))
    except ApiException as err:
        if err.body is not None:
            errbody = json.loads(err.body)
            click.echo("[Api Error] {}".format(errbody["message"]))
        else:
            click.echo("[Api Error] {}".format(err))


@click.command("experiment")
@click.argument("id")
@click.option("--wait/--no-wait", is_flag=True, default=False)
def delete_experiment(id, wait):
    """Delete experiment"""
    console = Console()
    try:
        thread = experimentClient.delete_experiment_async(id)
        timeout = time.time() + TIMEOUT
        with console.status("[bold green] Deleting Experiment(id = {} )...".format(id)):
            while not thread.ready():
                time.sleep(POLLING_INTERVAL)
                if time.time() > timeout:
                    console.print("[bold red] Timeout!")
                    return

        result = thread.get()
        result = result.result

        if wait:
            if result["status"] == "Deleted":
                console.print("[bold green] Experiment(id = {} ) deleted".format(id))
            else:
                console.print("[bold red] Failed")
                json_data = richJSON.from_data(result)
                console.print(Panel(json_data, title="Experiment(id = {} )".format(id)))

    except ApiException as err:
        if err.body is not None:
            errbody = json.loads(err.body)
            click.echo("[Api Error] {}".format(errbody["message"]))
        else:
            click.echo("[Api Error] {}".format(err))
