---
title: Submarine CLI
---

<!--
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
-->

Submarine CLI comes with pysubmarine python package. You can get CLI tools by pip installing apache-submarine.

## Config

You can set your CLI settings by this command

## Init

```bash
submarine config init 
```
Return
```bash
Submarine CLI Config initialized
```

Restore CLI config to default (hostname=`localhost`,port=`32080`)

## Show current config

```bash
submarine config list 
```
For example : return
```bash
╭──────────────────── SubmarineCliConfig ─────────────────────╮
│ {                                                           │
│   "connection": {                                           │
│     "hostname": "localhost",                                │
│     "port": 32080                                           │
│   }                                                         │
│ }                                                           │
╰─────────────────────────────────────────────────────────────╯
```

## Set config

```bash
submarine config set <parameter_path> <value> 
```

For example,
Set connection port to 8080:
```bash
submarine config set connection.port 8080
```

## Get config

```bash
submarine config get <parameter_path>
```

For example,
```bash
submarine config get connection.port
```
Return
```bash
connection.port=8080
```

## Notebooks

### List Notebooks

```bash
submarine list notebook 
```

### Get Notebooks

```bash
submarine get notebook <notebook id>
```

> you can get notebook id by using `list command`

### Delete Notebooks

```bash
submarine delete notebook <notebook id>
```

## Experiments

### List Experiments

```bash
submarine list experiment 
```

### Get Experiment

```bash
submarine get experiment <experiment id>
```

> you can get experiment id by using `list command`

### Delete Experiment

```bash
submarine delete experiment <experiment id> [--wait/--no-wait]
```
* --wait/--no-wait: blocking or non blocking (default no wait)

## Environments

### List Environments

```bash
submarine list environment 
```

### Get Environments

```bash
submarine get environment <environment name>
```

### Delete Environments

```bash
submarine delete experiment <environment name>
```
