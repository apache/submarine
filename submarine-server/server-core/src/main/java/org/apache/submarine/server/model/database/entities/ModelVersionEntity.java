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

package org.apache.submarine.server.model.database.entities;

import java.sql.Timestamp;
import java.util.List;

public class ModelVersionEntity {
  private String name;

  private Integer version;

  private String userId;

  private String experimentId;

  private String modelType;

  private String currentStage;

  private Timestamp creationTime;

  private Timestamp lastUpdatedTime;

  private String dataset;

  private String description;

  private List<String> tags;

  public String getName() {
    return name;
  }

  public void setName(String name) {
    this.name = name;
  }

  public Integer getVersion() {
    return version;
  }

  public void setVersion(Integer version) {
    this.version = version;
  }

  public String getUserId() {
    return userId;
  }

  public void setUserId(String userId) {
    this.userId = userId;
  }

  public String getExperimentId() {
    return experimentId;
  }

  public void setExperimentId(String experimentId) {
    this.experimentId = experimentId;
  }

  public String getModelType() {
    return modelType;
  }

  public void setModelType(String modelType) {
    this.modelType = modelType;
  }

  public String getCurrentStage() {
    return currentStage;
  }

  public void setCurrentStage(String currentStage) {
    this.currentStage = currentStage;
  }

  public Timestamp getCreationTime() {
    return creationTime;
  }

  public void setCreationTime(Timestamp creationTime) {
    this.creationTime = creationTime;
  }

  public Timestamp getLastUpdatedTime() {
    return lastUpdatedTime;
  }

  public void setLastUpdatedTime(Timestamp lastUpdatedTime) {
    this.lastUpdatedTime = lastUpdatedTime;
  }

  public String getDataset() {
    return dataset;
  }

  public void setDataset(String dataset) {
    this.dataset = dataset;
  }

  public String getDescription() {
    return description;
  }

  public void setDescription(String description) {
    this.description = description;
  }

  public List<String> getTags() {
    return tags;
  }

  public void setTags(List<String> tags) {
    this.tags = tags;
  }

  public ModelVersionEntity() {}

  public String toString() {
    return "ModelVersionEntity{" +
      "name='" + name + '\'' +
      ",version='" + version + '\'' +
      ", userId='" + userId + '\'' +
      ", experimentId='" + experimentId + '\'' +
      ", modelType='" + modelType + '\'' +
      ", currentStage='" + currentStage + '\'' +
      ", creationTime='" + creationTime + '\'' +
      ", lastUpdatedTime=" + lastUpdatedTime + '\'' +
      ", dataset=" + dataset + '\'' +
      ", description='" + description + '\'' +
      ", tags='" + tags + '\'' +
      '}';
  }
}
