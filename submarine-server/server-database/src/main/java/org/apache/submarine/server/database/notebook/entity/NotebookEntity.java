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

package org.apache.submarine.server.database.notebook.entity;

import java.util.Date;

import org.apache.submarine.server.database.database.entity.BaseEntity;
import org.apache.submarine.server.database.workbench.utils.CustomJsonDateDeserializer;

import com.fasterxml.jackson.databind.annotation.JsonDeserialize;

public class NotebookEntity extends BaseEntity {
  /*
    Take id (inherited from BaseEntity) as the primary key for notebook table
  */
  private String notebookSpec;
  
  private String notebookStatus;

  private String notebookUrl;
  private String reason;
  @JsonDeserialize(using = CustomJsonDateDeserializer.class)
  private Date deletedTime = new Date();


  public NotebookEntity() {
  }

  public String getNotebookSpec() {
    return notebookSpec;
  }

  public void setNotebookSpec(String notebookSpec) {
    this.notebookSpec = notebookSpec;
  }
 
  public String getNotebookStatus() {
    return notebookStatus;
  }

  public void setNotebookStatus(String noteStatus) {
    this.notebookStatus = noteStatus;
  }


  public String getNotebookUrl() {
    return notebookUrl;
  }

  public void setNotebookUrl(String notebookUrl) {
    this.notebookUrl = notebookUrl;
  }

  public String getReason() {
    return reason;
  }

  public void setReason(String reason) {
    this.reason = reason;
  }

  public Date getDeletedTime() {
    return deletedTime;
  }

  public void setDeletedTime(Date deletedTime) {
    this.deletedTime = deletedTime;
  }

  @Override
  public String toString() {
    return "NotebookEntity{" +
        "notebookSpec='" + notebookSpec + '\'' +
        ", id='" + id + '\'' +
        ", createBy='" + createBy + '\'' +
        ", createTime=" + createTime +
        ", updateBy='" + updateBy + '\'' +
        ", updateTime=" + updateTime + '\'' +
        ", notebookStatus='" + notebookStatus + "\'" +
        ", notebookUrl= '" + notebookUrl + "\'" +
        ", reason= '" + reason + "\'" +
        ", deletedTime= '" + deletedTime + "\'" +
        '}';
  }
}
