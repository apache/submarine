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

# How to generate Experiment API

### Build Submarine project
- git clone git@github.com:apache/submarine.git
- mvn clean package -DskipTests
### Start Submarine server
- cd submarine-dist/target/submarine-dist-0.4.0-SNAPSHOT-hadoop-2.9/submarine-dist-0.4.0-SNAPSHOT-hadoop-2.9/
- ./bin/submarine-daemon.sh start getMysqlJar
### Generate experiment API
- open localhost:8080/v1/openapi.json
- copy `openapi.json` file to [swagger editor](https://editor.swagger.io/)
- click *generate client* -> *python* to generate experiment API archive
### Add experiment API to pysubmarine
- mv `./python-client-generated/swagger-client/*` to `pysubmarine/submarine/job`
- rename all `swagger_client` in `submarine/job/*.py` to `submarine.job`.
    - e.g. `from swagger_client.models.job_task_spec import JobTaskSpec` -> `from submarine.job.models.job_task_spec import JobTaskSpec`
- import experiment API in [\_\_init__.py](../__init__.py)
