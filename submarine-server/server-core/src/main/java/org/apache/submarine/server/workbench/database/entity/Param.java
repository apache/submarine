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

import org.apache.submarine.server.database.entity.BaseEntity;

/*
# +-----------------------+----------+-------+--------------+
# | id                    | key      | value | worker_index |
# +-----------------------+----------+-------+--------------+
# | application_123456898 | max_iter | 100   | worker-1     |
# | application_123456898 | alpha    | 10    | worker-1     |
# | application_123456898 | n_jobs   | 5     | worker-1     |
# +-----------------------+----------+-------+--------------+
*/

public class Param extends BaseEntity {

  private String key;
  private String value;
  private String workerIndex;

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
}
