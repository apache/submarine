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
package org.apache.submarine.rest;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import org.apache.commons.lang.time.DateUtils;
import org.apache.submarine.database.entity.SysUser;
import org.apache.submarine.database.service.SysUserService;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.ws.rs.core.Response;

import java.util.Date;
import java.util.List;

import static junit.framework.TestCase.assertEquals;
import static org.junit.Assert.assertTrue;

public class SysUserRestApiTest extends CommonDataTest {
  private static final Logger LOG = LoggerFactory.getLogger(SysUserRestApiTest.class);

  private static SysUserRestApi userRestApi = new SysUserRestApi();
  private static SysUserService userService = new SysUserService();

  private static GsonBuilder gsonBuilder = new GsonBuilder();
  private static Gson gson = gsonBuilder.setDateFormat("yyyy-MM-dd HH:mm:ss").create();

  @Test
  public void editUserTest() throws Exception {
    SysUser sysUser = new SysUser();
    sysUser.setId(CommonDataTest.userId);
    sysUser.setUserName("user_name_update");
    sysUser.setRealName("real_name_update");
    sysUser.setPassword("password_update");
    sysUser.setAvatar("avatar_update");
    sysUser.setDeleted(9);
    sysUser.setPhone("123456789_update");
    sysUser.setRoleCode("roleCode_update");
    sysUser.setSex("SYS_USER_SEX_FEMALE");
    sysUser.setStatus("SYS_USER_STATUS_LOCKED");
    sysUser.setEmail("test@submarine.org_update");
    sysUser.setBirthday(new Date());
    sysUser.setDeptCode("A");
    sysUser.setBirthday(new Date());
    sysUser.setCreateTime(new Date());
    sysUser.setUpdateTime(new Date());

    Response response = userRestApi.edit(sysUser);
    CommonDataTest.assertUserResponseSuccess(response);

    List<SysUser> userList = userService.queryPageList("", null, null, null, null, 0, 10);
    assertTrue(userList.size() > 0);

    SysUser checkUser = userList.get(0);
    assertEquals(sysUser.getUserName(), checkUser.getUserName());
    assertEquals(sysUser.getRealName(), checkUser.getRealName());
    assertEquals(sysUser.getStatus(), checkUser.getStatus());
    assertEquals(sysUser.getSex(), checkUser.getSex());
    assertEquals(sysUser.getRoleCode(), checkUser.getRoleCode());
    assertEquals(sysUser.getPhone(), checkUser.getPhone());
    assertEquals(sysUser.getPassword(), checkUser.getPassword());
    // NOTE: dept_name value is left join query
    assertEquals(checkUser.getDeptName(), "deptA");
    assertEquals(sysUser.getDeleted(), checkUser.getDeleted());
    assertTrue(DateUtils.isSameDay(sysUser.getBirthday(), checkUser.getBirthday()));
    assertTrue(DateUtils.isSameDay(sysUser.getCreateTime(), checkUser.getCreateTime()));
    assertTrue(DateUtils.isSameDay(sysUser.getUpdateTime(), checkUser.getUpdateTime()));
    assertEquals(sysUser.getAvatar(), checkUser.getAvatar());
    assertEquals(sysUser.getEmail(), checkUser.getEmail());

    CommonDataTest.clearUserTable();
  }
}
