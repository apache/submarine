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

public class ModelVersionEntity {
  private String name;

  private Integer version;

  private Long createTime;

  private Long lastUpdatedTime;

  private String description;

  private String userId;

  private String currentStage;

  private String source;

  private String runId;

  private String status;

  private String statusMessage;

  private String runLink;

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

  public Long getCreateTime() {
    return createTime;
  }

  public void setCreateTime(Long createTime) {
    this.createTime = createTime;
  }

  public Long getLastUpdatedTime() {
    return lastUpdatedTime;
  }

  public void setLastUpdatedTime(Long lastUpdatedTime) {
    this.lastUpdatedTime = lastUpdatedTime;
  }

  public String getDescription() {
    return description;
  }

  public void setDescription(String description) {
    this.description = description;
  }

  public String getUserId() {
    return userId;
  }

  public void setUserId(String userId) {
    this.userId = userId;
  }

  public String getCurrentStage() {
    return currentStage;
  }

  public void setCurrentStage(String currentStage) {
    this.currentStage = currentStage;
  }

  public String getSource() {
    return source;
  }

  public void setSource(String source) {
    this.source = source;
  }

  public String getRunId() {
    return runId;
  }

  public void setRunId(String runId) {
    this.runId = runId;
  }

  public String getStatus() {
    return status;
  }

  public void setStatus(String status) {
    this.status = status;
  }

  public String getStatusMessage() {
    return statusMessage;
  }

  public void setStatusMessage(String statusMessage) {
    this.statusMessage = statusMessage;
  }

  public String getRunLink() {
    return runLink;
  }

  public void setRunLink(String runLink) {
    this.runLink = runLink;
  }

  public String toString() {
    return "ModelVersionEntity{" +
        "name='" + name + '\'' +
        ",version='" + version + '\'' +
        ", createTime='" + createTime + '\'' +
        ", lastUpdatedTime=" + lastUpdatedTime + '\'' +
        ", description='" + description + '\'' +
        ", userId='" + userId + '\'' +
        ", currentStage='" + currentStage + '\'' +
        ", source='" + source + '\'' +
        ", runLink='" + runLink + '\'' +
        ", runId='" + runId + '\'' +
        ", status='" + status + '\'' +
        ", statusMessage='" + statusMessage + '\'' +
        '}';
  }
}
