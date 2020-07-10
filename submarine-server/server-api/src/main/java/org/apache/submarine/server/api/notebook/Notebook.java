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

package org.apache.submarine.server.api.notebook;

import org.apache.submarine.server.api.spec.NotebookSpec;

/**
 * The notebook instance in submarine
 */
public class Notebook {
  private NotebookId notebookId;
  private String name;
  private String uid;
  private String url;
  private String status;
  private String createdTime;
  private String deletedTime;
  private NotebookSpec spec;

  public NotebookId getNotebookId() {
    return notebookId;
  }

  public void setNotebookId(NotebookId notebookId) {
    this.notebookId = notebookId;
  }

  public String getName() {
    return name;
  }

  public void setName(String name) {
    this.name = name;
  }

  public String getUid() {
    return uid;
  }

  public void setUid(String uid) {
    this.uid = uid;
  }

  public String getUrl() {
    return url;
  }

  public void setUrl(String url) {
    this.url = url;
  }

  public String getStatus() {
    return status;
  }

  public void setStatus(String status) {
    this.status = status;
  }

  public String getCreatedTime() {
    return createdTime;
  }

  public void setCreatedTime(String createdTime) {
    this.createdTime = createdTime;
  }

  public String getDeletedTime() {
    return deletedTime;
  }

  public void setDeletedTime(String deletedTime) {
    this.deletedTime = deletedTime;
  }

  public NotebookSpec getSpec() {
    return spec;
  }

  public void setSpec(NotebookSpec spec) {
    this.spec = spec;
  }

  public enum Status {
    STATUS_CREATED("Created"),
    STATUS_DELETED("Deleted");

    private String value;
    Status(String value) {
      this.value = value;
    }

    public String getValue() {
      return value;
    }

    @Override
    public String toString() {
      return value;
    }
  }

}
