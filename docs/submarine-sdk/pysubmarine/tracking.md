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

# pysubmarine-tracking
- Allow data scientist to track distributed ML job
- Support store ML parameters and metrics in Submarine-server
- [TODO] Support store ML job output (e.g. csv,images)
- Support hdfs,S3 and mysql (Currently we only support mysql)
- [TODO] (WEB) Metric tracking ui in workbench-web
- [TODO] (WEB) Metric graphical display in workbench-web

## Quickstart
1. [Start mini-submarine](../../../dev-support/mini-submarine#run-mini-submarine-image)

2. [Start Mysql server in mini-submarine](../../../dev-support/mini-submarine#run-workbench-server)

3. Uncomment the log_param and log_metric in
[mnist_distributed.py](../../../dev-support/mini-submarine/submarine/mnist_distributed.py)

4. Start Submarine job (e.g., run_submarine_mnist_tony.sh)

## Functions
### `submarine.get_tracking_uri()`

return the tracking URI.

### `submarine.set_tracking_uri(URI)`

set the tracking URI. You can also set the
SUBMARINE_TRACKING_URI environment variable to have Submarine find a URI from
there. The URI should be database connection string. 

**Parameters**

- URI - Submarine record data to Mysql server. The database URL
is expected in the format ``<dialect>+<driver>://<username>:<password>@<host>:<port>/<database>``.
By default it's `mysql+pymysql://submarine:password@localhost:3306/submarineDB`.
More detail : [SQLAlchemy docs](https://docs.sqlalchemy.org/en/latest/core/engines.html#database-urls)

<!--
    TODO : get database url from submarine-site.xml 
-->

### `submarine.log_metric(key, value, worker_index, step=0)`

logs a single key-value metric. The value must always be a number.

**Parameters**
- key - Metric name (string).
- value - Metric value (float).
- worker_index - The index of worker (string). Examples are "rank-1", "worker-2"
- step - A single integer step at which to log the specified Metrics,
by default it's 0.

### `submarine.log_param(key, value, worker_index)`

logs a single key-value parameter. The key and value are both strings.

**Parameters**
- key - Parameter name (string).
- value - Parameter value (string).
- worker_index - The index of worker (string). Examples are "rank-1", "worker-2"
