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
package org.apache.submarine.entity;

import java.util.List;

public class Role {
  private final String id;
  private final String name;
  private final String describe;
  private final int status;
  private final String creatorId;
  private final long createTime;
  private final int deleted;
  private final List<Permission> permissions;

  private Role(Builder builder) {
    this.id = builder.id;
    this.name = builder.name;
    this.describe = builder.describe;
    this.status = builder.status;
    this.creatorId = builder.creatorId;
    this.createTime = builder.createTime;
    this.deleted = builder.deleted;
    this.permissions = builder.permissions;
  }

  public static class Builder {
    private final String id;
    private final String name;

    private String describe;
    private int status;
    private String creatorId;
    private long createTime;
    private int deleted;
    private List<Permission> permissions;

    public Builder(String id, String name) {
      this.id = id;
      this.name = name;
    }

    public Builder describe(String describe) {
      this.describe = describe;
      return this;
    }

    public Builder status(int status) {
      this.status = status;
      return this;
    }

    public Builder creatorId(String creatorId) {
      this.creatorId = creatorId;
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

    public Builder permissions(List<Permission> permissions) {
      this.permissions = permissions;
      return this;
    }

    public Role build() {
      return new Role(this);
    }
  }

  @Override
  public String toString() {
    return "User{" +
        "id='" + id + '\'' +
        ", name='" + name + '\'' +
        ", describe=" + describe +
        ", status='" + status + '\'' +
        ", creatorId='" + creatorId + '\'' +
        ", createTime=" + createTime +
        ", creatorId='" + creatorId + '\'' +
        ", createTime='" + createTime + '\'' +
        ", deleted='" + deleted + '\'' +
        ", permissions='" + permissions.toString() + '\'' +
        '}';
  }
}
