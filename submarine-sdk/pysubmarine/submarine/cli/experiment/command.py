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

@click.command('experiment')
def list_experiment():
    """List experiments"""
    click.echo('list!')

@click.command('experiment')
@click.argument('id')
def get_experiment():
    """Get experiments"""
    click.echo('get! id={}'.format(id))

@click.command('experiment')
@click.argument('id')
def delete_experiment(id):
    """Delete experiment"""
    click.echo('delete! id={}'.format(id))