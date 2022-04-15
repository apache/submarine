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
package org.apache.submarine.server.database.workbench.entity;

import java.util.ArrayList;
import java.util.List;

import org.apache.submarine.server.database.database.entity.BaseEntity;

public class TeamEntity extends BaseEntity {

  private String owner;

  private String teamName;

  private List<TeamMemberEntity> collaborators = new ArrayList<>();;

  public String getOwner() {
    return owner;
  }

  public void setOwner(String owner) {
    this.owner = owner;
  }

  public String getTeamName() {
    return teamName;
  }

  public void setTeamName(String teamName) {
    this.teamName = teamName;
  }

  public List<TeamMemberEntity> getCollaborators() {
    return collaborators;
  }

  public void setCollaborators(List<TeamMemberEntity> collaborators) {
    this.collaborators = collaborators;
  }

  public void addCollaborator(TeamMemberEntity member) {
    this.collaborators.add(member);
  }
}
