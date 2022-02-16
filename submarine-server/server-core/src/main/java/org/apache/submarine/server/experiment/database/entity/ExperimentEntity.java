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

package org.apache.submarine.server.experiment.database.entity;

import org.apache.submarine.server.database.entity.BaseEntity;

public class ExperimentEntity extends BaseEntity {
  /*
    Take id (inherited from BaseEntity) as the primary key for experiment table
  */
  private String experimentSpec;

  private String experimentStatus;

  public ExperimentEntity() {}

  public String getExperimentSpec() {
    return experimentSpec;
  }

  public void setExperimentSpec(String experimentSpec) {
    this.experimentSpec = experimentSpec;
  }
 
  public String getExperimentStatus() {
    return experimentStatus;
  }

  public void setExperimentStatus(String experimentStatus) {
    this.experimentStatus = experimentStatus;
  }

  @Override
  public String toString() {
    return "ExperimentEntity{" +
      "experimentSpec='" + experimentSpec + '\'' +
      ", id='" + id + '\'' +
      ", createBy='" + createBy + '\'' +
      ", createTime=" + createTime +
      ", updateBy='" + updateBy + '\'' +
      ", updateTime='" + updateTime + '\'' +
      ", experimentStatus='" + experimentStatus + "\'" +
      '}';
  }
}
