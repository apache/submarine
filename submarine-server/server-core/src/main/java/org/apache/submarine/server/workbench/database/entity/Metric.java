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

import java.sql.Timestamp;

/*
# +--------------------+-------+-------------------+--------------+---------------+------+--------+
# | id                 | key   | value             | worker_index | timestamp     | step | is_nan |
# +--------------------+-------+-------------------+--------------+---------------+------+--------+
# | application_123456 | score | 0.666666666666667 | worker-1     | 1595414873838 |    0 |      0 |
# | application_123456 | score | 0.666666666666667 | worker-1     | 1595472286360 |    0 |      0 |
# | application_123456 | score | 0.666666666666667 | worker-1     | 1595414632967 |    0 |      0 |
# | application_123456 | score | 0.666666666666667 | worker-1     | 1595415075067 |    0 |      0 |
# +--------------------+-------+-------------------+--------------+---------------+------+--------+
*/

public class Metric extends BaseEntity {

  private String key;
  private Float value;
  private String workerIndex;
  private Timestamp timestamp;
  private Integer step;
  private Boolean isNan;

  public String getKey() {
    return this.key;
  }

  public void setKey(String key) {
    this.key = key;
  }

  public Float getValue() {
    return this.value;
  }

  public void setValue(Float value) {
    this.value = value;
  }

  public String getWorkerIndex() {
    return this.workerIndex;
  }

  public void setWorkerIndex(String workerIndex) {
    this.workerIndex = workerIndex;
  }

  public Timestamp getTimestamp() {
    return this.timestamp;
  }

  public void setTimestamp(Timestamp timestamp) {
    this.timestamp = timestamp;
  }

  public Integer getStep() {
    return this.step;
  }

  public void setStep(Integer step) {
    this.step = step;
  }

  public Boolean getIsNan() {
    return this.isNan;
  }

  public void setIsNan(Boolean isNan) {
    this.isNan = isNan;
  }
}
