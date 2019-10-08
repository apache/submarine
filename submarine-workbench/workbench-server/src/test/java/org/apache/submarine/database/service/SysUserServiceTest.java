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
package org.apache.submarine.database.service;

import org.apache.commons.lang.time.DateUtils;
import org.apache.submarine.database.entity.SysUser;
import org.junit.After;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Date;
import java.util.List;

import static junit.framework.TestCase.assertEquals;
import static org.junit.Assert.assertTrue;

public class SysUserServiceTest {
  private static final Logger LOG = LoggerFactory.getLogger(SysUserServiceTest.class);

  private SysUserService userService = new SysUserService();

  @After
  public void removeAllRecord() throws Exception {
    SysUserService userService = new SysUserService();

    List<SysUser> userList = userService.queryPageList("", null, null, null, null, 0, 10);
    assertTrue(userList.size() > 0);
    for (SysUser user : userList) {
      userService.delete(user.getId());
    }
  }

  @Test
  public void addUserTest() throws Exception {
    SysUser sysUser = new SysUser();
    sysUser.setUserName("user_name");
    sysUser.setRealName("real_name");
    sysUser.setPassword("password");
    sysUser.setAvatar("avatar");
    sysUser.setDeleted(1);
    sysUser.setPhone("123456789");
    sysUser.setRoleCode("roleCode");
    // sysUser.setSex("SYS_USER_SEX_MALE");
    // sysUser.setStatus("SYS_USER_STATUS_AVAILABLE");
    sysUser.setEmail("test@submarine.org");
    sysUser.setBirthday(new Date());
    // sysUser.setDeptCode("A");
    sysUser.setCreateTime(new Date());
    sysUser.setUpdateTime(new Date());

    Boolean ret = userService.add(sysUser);
    assertTrue(ret);

    List<SysUser> userList = userService.queryPageList(sysUser.getUserName(), null, null, null, null, 0, 10);
    LOG.debug("userList.size():{}", userList.size());
    assertEquals(userList.size(), 1);
    SysUser user = userList.get(0);

    assertEquals(sysUser.getEmail(), user.getEmail());
    assertEquals(sysUser.getToken(), user.getToken());
    assertEquals(sysUser.getAvatar(), user.getAvatar());
    assertTrue(DateUtils.isSameDay(sysUser.getBirthday(), user.getBirthday()));

    assertEquals(sysUser.getDeleted(), user.getDeleted());
    assertEquals(sysUser.getDeptCode(), user.getDeptCode());

    // assertNotNull(user.getDeptName());
    assertEquals(sysUser.getPassword(), user.getPassword());
    assertEquals(sysUser.getPhone(), user.getPhone());
    assertEquals(sysUser.getRealName(), user.getRealName());
    assertEquals(sysUser.getRoleCode(), user.getRoleCode());
    assertEquals(sysUser.getSex(), user.getSex());
    assertEquals(sysUser.getStatus(), user.getStatus());
    assertEquals(sysUser.getUserName(), user.getUserName());
    assertEquals(sysUser.getCreateBy(), user.getCreateBy());
    assertTrue(DateUtils.isSameDay(sysUser.getCreateTime(), user.getCreateTime()));
    assertEquals(sysUser.getId(), user.getId());
    assertEquals(sysUser.getUpdateBy(), user.getUpdateBy());
    assertTrue(DateUtils.isSameDay(sysUser.getUpdateTime(), user.getUpdateTime()));
  }

  @Test
  public void updateUserTest() throws Exception {
    SysUser sysUser = new SysUser();
    sysUser.setUserName("update_user_name");
    sysUser.setRealName("update_real_name");
    sysUser.setPassword("update_password");
    sysUser.setAvatar("update_avatar");
    sysUser.setDeleted(1);
    sysUser.setPhone("123456789");
    sysUser.setRoleCode("roleCode");
    // sysUser.setSex("SYS_USER_SEX_MALE");
    // sysUser.setStatus("SYS_USER_STATUS_AVAILABLE");
    sysUser.setEmail("test@submarine.org");
    sysUser.setBirthday(new Date());
    // sysUser.setDeptCode("A");
    sysUser.setCreateTime(new Date());
    sysUser.setUpdateTime(new Date());

    Boolean ret = userService.add(sysUser);
    assertTrue(ret);

    // update sys user
    SysUser updateUser = new SysUser();
    updateUser.setId(sysUser.getId());
    updateUser.setUserName(sysUser.getUserName() + "_1");
    updateUser.setUserName(sysUser.getUserName() + "_1");
    updateUser.setRealName(sysUser.getRealName() + "_1");
    updateUser.setPassword(sysUser.getPassword() + "_1");
    updateUser.setAvatar(sysUser.getAvatar() + "_1");
    updateUser.setDeleted(2);
    updateUser.setPhone(sysUser.getPhone() + "_1");
    updateUser.setRoleCode(sysUser.getRoleCode() + "_1");
    // updateUser.setSex("SYS_USER_SEX_FEMALE");
    // updateUser.setStatus("SYS_USER_STATUS_LOCKED");
    updateUser.setEmail(sysUser.getEmail() + "_1");
    updateUser.setBirthday(new Date());
    // updateUser.setDeptCode("AA");
    updateUser.setCreateTime(new Date());
    updateUser.setUpdateTime(new Date());

    ret = userService.edit(updateUser);
    assertTrue(ret);

    List<SysUser> userList = userService.queryPageList(
        updateUser.getUserName(), null, null, null, null, 0, 10);
    assertEquals(userList.size(), 1);
    SysUser user = userList.get(0);

    assertEquals(updateUser.getEmail(), user.getEmail());
    assertEquals(updateUser.getToken(), user.getToken());
    assertEquals(updateUser.getAvatar(), user.getAvatar());
    assertEquals(updateUser.getDeleted(), user.getDeleted());
    assertEquals(updateUser.getDeptCode(), user.getDeptCode());
    assertEquals(updateUser.getPassword(), user.getPassword());
    assertEquals(updateUser.getPhone(), user.getPhone());
    assertEquals(updateUser.getRealName(), user.getRealName());
    assertEquals(updateUser.getRoleCode(), user.getRoleCode());
    assertEquals(updateUser.getSex(), user.getSex());
    assertEquals(updateUser.getStatus(), user.getStatus());
    assertEquals(updateUser.getUserName(), user.getUserName());
    assertEquals(updateUser.getCreateBy(), user.getCreateBy());
    assertEquals(updateUser.getId(), user.getId());
    assertEquals(updateUser.getUpdateBy(), user.getUpdateBy());

    // assertNotNull(user.getDeptName());
    assertTrue(DateUtils.isSameDay(updateUser.getBirthday(), user.getBirthday()));
    assertTrue(DateUtils.isSameDay(updateUser.getCreateTime(), user.getCreateTime()));
    assertTrue(DateUtils.isSameDay(updateUser.getUpdateTime(), user.getUpdateTime()));
  }
}
