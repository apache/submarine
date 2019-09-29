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

# Submarine-SDK

Support Python, Scala, R language for algorithm development.
The SDK is provided to help developers use submarine's internal data caching, 
data exchange, and task tracking to more efficiently improve the development 
and execution of machine learning tasks.

- Allow data scients to track distributed ML job 
- Support store ML parameters and metrics in Submarine-server
- Support store ML job output (e.g. csv,images)
- Support hdfs,S3 and mysql 
- (WEB) Metric tracking ui in workbench-web
- (WEB) Metric graphical display in workbench-web

### Project setup
- Clone repo
```bash
git https://github.com/apache/hadoop-submarine.git
cd hadoop-submarine/submarine-sdk
```

- Install pip package
```
pip install .
```

- Run tests
```
pytest --cov=submarine -vs
```

- Run checkstyle
```
pylint --msg-template="{path} ({line},{column}): \
[{msg_id} {symbol}] {msg}" --rcfile=pylintrc -- submarine tests
```