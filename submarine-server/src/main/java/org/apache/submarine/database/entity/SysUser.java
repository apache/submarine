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

import java.util.Date;

public class SysUser extends BaseEntity {
  private String name;
  private String username;
  private String password;
  private String avatar;
  private Integer sex;
  private Integer status;
  private String phone;
  private String email;
  private String org_code;
  private String lastLoginIp;
  private Date lastLoginTime;
  private Integer deleted;
  private String roleId;
  private String lang;
  private String token;

  public void setUsername(String username) {
    this.username = username;
  }

  public void setPassword(String password) {
    this.password = password;
  }

  public void setAvatar(String avatar) {
    this.avatar = avatar;
  }

  public void setSex(Integer sex) {
    this.sex = sex;
  }

  public void setStatus(Integer status) {
    this.status = status;
  }

  public void setPhone(String phone) {
    this.phone = phone;
  }

  public void setEmail(String email) {
    this.email = email;
  }

  public void setOrgCode(String orgCode) {
    this.org_code = orgCode;
  }

  public void setLastLoginIp(String lastLoginIp) {
    this.lastLoginIp = lastLoginIp;
  }

  public void setLastLoginTime(Date lastLoginTime) {
    this.lastLoginTime = lastLoginTime;
  }

  public void setDeleted(Integer deleted) {
    this.deleted = deleted;
  }

  public void setRoleId(String roleId) {
    this.roleId = roleId;
  }

  public void setLang(String lang) {
    this.lang = lang;
  }

  public void setToken(String token) {
    this.token = token;
  }

  public void setName(String name) {
    this.name = name;
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
        ", phone='" + phone + '\'' +
        ", email='" + email + '\'' +
        ", org_code=" + org_code +
        ", lastLoginIp=" + lastLoginIp +
        ", lastLoginTime=" + lastLoginTime +
        ", deleted='" + deleted + '\'' +
        ", createBy='" + create_by + '\'' +
        ", createTime='" + create_time + '\'' +
        ", updateBy='" + update_by + '\'' +
        ", updateTime='" + update_time + '\'' +
        ", roleId='" + roleId + '\'' +
        ", lang='" + lang + '\'' +
        ", token='" + token + '\'' +
        '}';
  }
}

