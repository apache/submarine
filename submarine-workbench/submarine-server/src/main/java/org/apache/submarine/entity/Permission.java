/**
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * <p>
 * http://www.apache.org/licenses/LICENSE-2.0
 * <p>
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. See accompanying LICENSE file.
 */
package org.apache.submarine.entity;

import java.util.List;

public class Permission {
  private final String roleId;
  private final String permissionId;
  private final String permissionName;
  private final String dataAccess;
  private final List<Action> actionList;
  private final List<Action> actions;
  private final List<Action> actionEntitySet;

  private Permission(Builder builder) {
    this.roleId = builder.roleId;
    this.permissionId = builder.permissionId;
    this.permissionName = builder.permissionName;
    this.dataAccess = builder.dataAccess;
    this.actionList = builder.actionList;
    this.actions = builder.actions;
    this.actionEntitySet = builder.actionEntitySet;
  }

  public static class Builder {
    private final String roleId;
    private final String permissionId;
    private final String permissionName;

    private String dataAccess;
    private List<Action> actionList;
    private List<Action> actions;
    private List<Action> actionEntitySet;

    public Builder(String roleId, String permissionId, String permissionName) {
      this.roleId = roleId;
      this.permissionId = permissionId;
      this.permissionName = permissionName;
    }

    public Builder dataAccess(String dataAccess) {
      this.dataAccess = dataAccess;
      return this;
    }

    public Builder actionList(List<Action> actionList) {
      this.actionList = actionList;
      return this;
    }

    public Builder actions(List<Action> actions) {
      this.actions = actions;
      return this;
    }

    public Builder actionEntitySet(List<Action> actionEntitySet) {
      this.actionEntitySet = actionEntitySet;
      return this;
    }

    public Permission build() {
      return new Permission(this);
    }
  }

  @Override
  public String toString() {
    return "User{" +
        "roleId='" + roleId + '\'' +
        ", permissionId='" + permissionId + '\'' +
        ", permissionName=" + permissionName +
        ", dataAccess='" + dataAccess + '\'' +
        ", actionList=" + actionList.toString() +
        ", actions='" + actions.toString() + '\'' +
        ", actionEntitySet='" + actionEntitySet.toString() + '\'' +
        '}';
  }
}
