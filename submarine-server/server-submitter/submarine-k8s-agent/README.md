<!---
  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License. See accompanying LICENSE file.
-->

# Submarine Agent

## Development Guide

The `submarine-server` created with operator contains a `SUBMARINE_UID` environment variable,
which we will also need to configure locally during the development phase.
```shell
export SUBMARINE_UID=${submarine_uid}
```

Also, we need use `port-forward` to link the database port to a local connection
```shell
kubectl port-forward service/submarine-database 3306:3306 -n submarine-user-test
```

## Build Image

We already have a script to automate the image build
```shell
cd ./dev-support/docker-images/agent
./build.sh
```
