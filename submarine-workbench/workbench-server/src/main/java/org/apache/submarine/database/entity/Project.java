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
package org.apache.submarine.database.entity;

import java.util.ArrayList;
import java.util.List;

public class Project extends BaseEntity {
  private String projectName;

  // 0:Private, 1:Team, 2:Public
  private Integer visibility;

  // 0:notebook, 1:python, 2:spark, 3:R, 4:tensorflow, 5:pytorch
  private Integer type;

  private String description;

  private String userName;

  private List<ProjectFiles> projectFilesList;

  public String getProjectName() {
    return projectName;
  }

  public void setProjectName(String projectName) {
    this.projectName = projectName == null ? null : projectName.trim();
  }

  public Integer getVisibility() {
    return visibility;
  }

  public void setVisibility(Integer visibility) {
    this.visibility = visibility;
  }

  public Integer getType() {
    return type;
  }

  public void setType(Integer type) {
    this.type = type;
  }

  public String getDescription() {
    return description;
  }

  public void setDescription(String description) {
    this.description = description == null ? null : description.trim();
  }

  public String getUserName() {
    return userName;
  }

  public void setUserName(String userName) {
    this.userName = userName == null ? null : userName.trim();
  }

  public List<ProjectFiles> getProjectFilesList() {
    return projectFilesList;
  }

  public void setProjectFilesList(List<ProjectFiles> projectFilesList) {
    this.projectFilesList = projectFilesList;
  }

  public void addProjectFilesList(ProjectFiles projectFiles) {
    if (projectFilesList == null) {
      projectFilesList = new ArrayList<>();
    }
    this.projectFilesList.add(projectFiles);
  }
}
