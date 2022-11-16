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
package org.apache.submarine.server.api.workbench;

public class UserInfo {
  private final String id;
  private final String name;
  private final String username;
  private final String password;
  private final String avatar;
  private final String status;
  private final String telephone;
  private final String lastLoginIp;
  private final long lastLoginTime;
  private final String creatorId;
  private final long createTime;
  private final String merchantCode;
  private final int deleted;
  private final String roleId;
  private final Role role;

  private UserInfo(UserInfo.Builder builder) {
    this.id = builder.id;
    this.name = builder.name;
    this.username = builder.username;
    this.password = builder.password;
    this.avatar = builder.avatar;
    this.status = builder.status;
    this.telephone = builder.telephone;
    this.lastLoginIp = builder.lastLoginIp;
    this.lastLoginTime = builder.lastLoginTime;
    this.creatorId = builder.creatorId;
    this.createTime = builder.createTime;
    this.deleted = builder.deleted;
    this.roleId = builder.roleId;
    this.merchantCode = builder.merchantCode;
    this.role = builder.role;
  }

  public static class Builder {
    private final String id;
    private final String name;

    private String username;
    private String password;
    private String avatar;
    private String status;
    private String telephone;
    private String lastLoginIp;
    private long lastLoginTime;
    private String creatorId;
    private long createTime;
    private String merchantCode;
    private int deleted = 0;
    private String roleId;
    private Role role;

    public Builder(String id, String name) {
      this.id = id;
      this.name = name;
    }

    public Builder username(String username) {
      this.username = username;
      return this;
    }

    public Builder password(String password) {
      this.password = password;
      return this;
    }

    public Builder avatar(String avatar) {
      this.avatar = avatar;
      return this;
    }

    public Builder status(String status) {
      this.status = status;
      return this;
    }

    public Builder lastLoginIp(String lastLoginIp) {
      this.lastLoginIp = lastLoginIp;
      return this;
    }

    public Builder lastLoginTime(long lastLoginTime) {
      this.lastLoginTime = lastLoginTime;
      return this;
    }

    public Builder creatorId(String creatorId) {
      this.creatorId = creatorId;
      return this;
    }

    public Builder telephone(String telephone) {
      this.telephone = telephone;
      return this;
    }

    public Builder createTime(long createTime) {
      this.createTime = createTime;
      return this;
    }

    public Builder deleted(int deleted) {
      this.deleted = deleted;
      return this;
    }

    public Builder roleId(String roleId) {
      this.roleId = roleId;
      return this;
    }

    public Builder merchantCode(String merchantCode) {
      this.merchantCode = merchantCode;
      return this;
    }

    public Builder role(Role role) {
      this.role = role;
      return this;
    }

    public UserInfo build() {
      return new UserInfo(this);
    }
  }

  @Override
  public String toString() {
    return "User{" +
        "id='" + id + '\'' +
        ", name='" + name + '\'' +
        ", username=" + username +
        ", password='" + password + '\'' +
        ", avatar=" + avatar +
        ", status='" + status + '\'' +
        ", telephone='" + telephone + '\'' +
        ", lastLoginIp=" + lastLoginIp +
        ", creatorId='" + creatorId + '\'' +
        ", createTime='" + createTime + '\'' +
        ", deleted='" + deleted + '\'' +
        ", roleId='" + roleId + '\'' +
        ", lang='" + merchantCode + '\'' +
        ", role='" + role.toString() + '\'' +
        '}';
  }
}
