/**
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. See accompanying LICENSE file.
 */
package org.apache.submarine.database.entity;

import java.util.ArrayList;
import java.util.List;

public class Team extends BaseEntity {

  private String owner;

  private String teamName;

  private List<TeamMember> collaborators;

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

  public List<TeamMember> getCollaborators() {
    return collaborators;
  }

  public void setCollaborators(List<TeamMember> collaborators) {
    this.collaborators = collaborators;
  }

  public void addCollaborator(TeamMember memeber) {
    if (collaborators == null) {
      collaborators = new ArrayList<>();
    }
    this.collaborators.add(memeber);
  }
}
