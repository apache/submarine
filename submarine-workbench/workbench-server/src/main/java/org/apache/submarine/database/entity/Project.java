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

import org.apache.submarine.annotation.Dict;

import java.util.ArrayList;
import java.util.List;

public class Project extends BaseEntity {
  private String name;

  @Dict(Code = "PROJECT_VISIBILITY")
  private String visibility;

  @Dict(Code = "PROJECT_TYPE")
  private String type;

  @Dict(Code = "PROJECT_PERMISSION")
  private String permission;

  // Comma separated tag
  private String tags;

  // number of star
  private Integer starNum = 0;

  // number of like
  private Integer likeNum = 0;

  // number of message
  private Integer messageNum = 0;

  // Team.teamName
  private String teamName;

  private String description;

  private String userName;

  private List<ProjectFiles> projectFilesList = new ArrayList<>();

  public String getName() {
    return name;
  }

  public void setName(String name) {
    this.name = name == null ? null : name.trim();
  }

  public String getVisibility() {
    return visibility;
  }

  public void setVisibility(String visibility) {
    this.visibility = visibility;
  }

  public String getType() {
    return type;
  }

  public void setType(String type) {
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
    this.projectFilesList.add(projectFiles);
  }

  public String getTags() {
    return tags;
  }

  public void setTags(String tags) {
    this.tags = tags;
  }

  public Integer getStarNum() {
    return starNum;
  }

  public void setStarNum(Integer starNum) {
    this.starNum = starNum;
  }

  public Integer getLikeNum() {
    return likeNum;
  }

  public void setLikeNum(Integer likeNum) {
    this.likeNum = likeNum;
  }

  public Integer getMessageNum() {
    return messageNum;
  }

  public void setMessageNum(Integer messageNum) {
    this.messageNum = messageNum;
  }

  public String getTeamName() {
    return teamName;
  }

  public void setTeamName(String teamName) {
    this.teamName = teamName;
  }

  public String getPermission() {
    return permission;
  }

  public void setPermission(String permission) {
    this.permission = permission;
  }
}
