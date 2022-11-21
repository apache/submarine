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

import com.fasterxml.jackson.databind.annotation.JsonDeserialize;

import org.apache.submarine.server.database.entity.BaseEntity;
import org.apache.submarine.server.database.workbench.utils.CustomJsonDateDeserializer;
import org.apache.submarine.server.database.workbench.annotation.Dict;

import java.util.Date;

public class SysUserEntity extends BaseEntity {
  private String userName;
  private String realName;
  private String password;
  private String avatar;

  @Dict(Code = "SYS_USER_SEX")
  private String sex;
  @Dict(Code = "SYS_USER_STATUS")
  private String status;
  private String phone;
  private String email;
  private String deptCode;
  private String deptName;

  private String roleCode;

  @JsonDeserialize(using = CustomJsonDateDeserializer.class)
  protected Date birthday = new Date();

  private Integer deleted;
  private String token;

  public String getUserName() {
    return userName;
  }

  public void setUserName(String userName) {
    this.userName = userName;
  }

  public String getRealName() {
    return realName;
  }

  public void setRealName(String realName) {
    this.realName = realName;
  }

  public String getPassword() {
    return password;
  }

  public void setPassword(String password) {
    this.password = password;
  }

  public String getAvatar() {
    return avatar;
  }

  public void setAvatar(String avatar) {
    this.avatar = avatar;
  }

  public String getSex() {
    return sex;
  }

  public void setSex(String sex) {
    this.sex = sex;
  }

  public String getStatus() {
    return status;
  }

  public void setStatus(String status) {
    this.status = status;
  }

  public String getPhone() {
    return phone;
  }

  public void setPhone(String phone) {
    this.phone = phone;
  }

  public String getEmail() {
    return email;
  }

  public void setEmail(String email) {
    this.email = email;
  }

  public String getDeptCode() {
    return deptCode;
  }

  public void setDeptCode(String deptCode) {
    this.deptCode = deptCode;
  }

  public String getRoleCode() {
    return roleCode;
  }

  public void setRoleCode(String roleCode) {
    this.roleCode = roleCode;
  }

  public Integer getDeleted() {
    return deleted;
  }

  public void setDeleted(Integer deleted) {
    this.deleted = deleted;
  }

  public String getToken() {
    return token;
  }

  public void setToken(String token) {
    this.token = token;
  }

  public String getDeptName() {
    return deptName;
  }

  public void setDeptName(String deptName) {
    this.deptName = deptName;
  }

  public Date getBirthday() {
    return birthday;
  }

  public void setBirthday(Date birthday) {
    this.birthday = birthday;
  }

  @Override
  public String toString() {
    return "SysUserEntity{" +
            "userName='" + userName + '\'' +
            ", realName='" + realName + '\'' +
            ", password='" + password + '\'' +
            ", avatar='" + avatar + '\'' +
            ", sex='" + sex + '\'' +
            ", status='" + status + '\'' +
            ", phone='" + phone + '\'' +
            ", email='" + email + '\'' +
            ", deptCode='" + deptCode + '\'' +
            ", deptName='" + deptName + '\'' +
            ", roleCode='" + roleCode + '\'' +
            ", birthday=" + birthday +
            ", deleted=" + deleted +
            ", token='" + token + '\'' +
            ", id='" + id + '\'' +
            ", createBy='" + createBy + '\'' +
            ", createTime=" + createTime +
            ", updateBy='" + updateBy + '\'' +
            ", updateTime=" + updateTime +
            '}';
  }
}
