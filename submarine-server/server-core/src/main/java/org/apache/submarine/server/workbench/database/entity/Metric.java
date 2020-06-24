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

import java.math.BigInteger;

/*
# +-------+----------+--------------+---------------+------+--------+------------------+
# | key   | value    | worker_index | timestamp     | step | is_nan | job_name         |
# +-------+----------+--------------+---------------+------+--------+------------------+
# | score | 0.666667 | worker-1     | 1569139525097 |    0 |      0 | application_1234 |
# | score | 0.666667 | worker-1     | 1569149139731 |    0 |      0 | application_1234 |
# | score | 0.666667 | worker-1     | 1569169376482 |    0 |      0 | application_1234 |
# | score | 0.666667 | worker-1     | 1569236290721 |    0 |      0 | application_1234 |
# | score | 0.666667 | worker-1     | 1569236466722 |    0 |      0 | application_1234 |
# +-------+----------+--------------+---------------+------+--------+------------------+
*/

public class Metric extends BaseEntity {

  private String key;
  private Float value;
  private String workerIndex;
  private BigInteger timestamp;
  private Integer step;
  private Integer isNan;
  private String jobName;

  public String getKey() {
    return this.key;
  }

  public void setKey(String metricKey) {
    this.key = metricKey;
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

  public BigInteger getTimestamp() {
    return this.timestamp;
  }

  public void setTimestamp(BigInteger timestamp) {
    this.timestamp = timestamp;
  }

  public void setTimestamp(Integer timestamp) {
    this.timestamp = BigInteger.valueOf(timestamp.longValue());
  }

  public Integer getStep() {
    return this.step;
  }

  public void setStep(Integer step) {
    this.step = step;
  }

  public Integer getIsNan() {
    return this.isNan;
  }

  public void setIsNan(Integer isNan) {
    this.isNan = isNan;
  }

  public String getJobName() {
    return this.jobName;
  }

  public void setJobName(String jobName) {
    this.jobName = jobName;
  }
}
