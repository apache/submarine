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

  private String metricKey;
  private float value;
  private String workerIndex;
  private BigInteger timestamp;
  private int step;
  private int isNan;
  private String jobName;

  public String getMetric_key() {
    return this.metricKey;
  }

  public void setMetric_key(String metric_key) {
    this.metricKey = metric_key;
  }

  public float getValue() {
    return this.value;
  }

  public void setValue(float value) {
    this.value = value;
  }

  public String getWorker_index() {
    return this.workerIndex;
  }

  public void setWorker_index(String worker_index) {
    this.workerIndex = worker_index;
  }

  public BigInteger getTimestamp() {
    return this.timestamp;
  }

  public void setTimestamp(BigInteger timestamp) {
    this.timestamp = timestamp;
  }

  public int getStep() {
    return this.step;
  }

  public void setStep(int step) {
    this.step = step;
  }

  public int getIs_nan() {
    return this.isNan;
  }

  public void setIs_nan(int is_nan) {
    this.isNan = is_nan;
  }

  public String getJob_name() {
    return this.jobName;
  }

  public void setJob_name(String job_name) {
    this.jobName = job_name;
  }

}
