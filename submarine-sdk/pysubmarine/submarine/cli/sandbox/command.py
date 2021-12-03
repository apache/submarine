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

@click.command('start')
@click.option('-v', '--version', 'version', help='Specify sandbox version',default='0.6.0')
def start_sandbox(version):
    """Start sandbox"""
    click.echo('start sandbox! version={}'.format(version))

@click.command('delete')
def delete_sandbox():
    """Delete sandbox"""
    click.echo('delete sandbox!')