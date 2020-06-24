/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
package org.apache.submarine.server.workbench.database.entity;

/*
# +----------+-------+--------------+-----------------------+
# | key      | value | worker_index | job_name              |
# +----------+-------+--------------+-----------------------+
# | max_iter | 100   | worker-1     | application_123651651 |
# | n_jobs   | 5     | worker-1     | application_123456898 |
# | alpha    | 20    | worker-1     | application_123456789 |
# +----------+-------+--------------+-----------------------+
*/

public class Param extends BaseEntity {
  String key;
  String value;
  String workerIndex;
  String jobName;

  public String getKey() {
    return this.key;
  }

  public void setKey(String key) {
    this.key = key;
  }

  public String getValue() {
    return this.value;
  }

  public void setValue(String value) {
    this.value = value;
  }

  public String getWorkerIndex() {
    return this.workerIndex;
  }

  public void setWorkerIndex(String workerIndex) {
    this.workerIndex = workerIndex;
  }

  public String getJobName() {
    return this.jobName;
  }

  public void setJobName(String jobName) {
    this.jobName = jobName;
  }
}
