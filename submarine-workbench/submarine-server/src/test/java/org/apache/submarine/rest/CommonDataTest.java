/**
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
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
import com.google.gson.reflect.TypeToken;
import org.apache.submarine.database.entity.SysDept;
import org.apache.submarine.database.entity.SysDeptTree;
import org.apache.submarine.database.entity.SysDict;
import org.apache.submarine.database.entity.SysDictItem;
import org.apache.submarine.database.entity.SysUser;
import org.apache.submarine.database.service.SysUserService;
import org.apache.submarine.server.JsonResponse;
import org.apache.submarine.server.JsonResponse.ListResult;
import org.junit.AfterClass;
import org.junit.Assert;
import org.junit.BeforeClass;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.ws.rs.core.Response;

import java.lang.reflect.Type;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class CommonDataTest {
  private static final Logger LOG = LoggerFactory.getLogger(CommonDataTest.class);

  private static GsonBuilder gsonBuilder = new GsonBuilder();
  private static Gson gson = gsonBuilder.setDateFormat("yyyy-MM-dd HH:mm:ss").create();

  private static SysDictRestApi dictRestApi = new SysDictRestApi();
  private static SysDictItemRestApi dictItemRestApi = new SysDictItemRestApi();
  private static SysDeptRestApi deptRestApi = new SysDeptRestApi();
  private static SysUserRestApi userRestApi = new SysUserRestApi();
  private static SysUserService userService = new SysUserService();

  public static String userId = "";

  @BeforeClass
  public static void startTest() {
    LOG.debug(">>> createAllTable()");
    createAllTable();
  }

  @AfterClass
  public static void exitTest() throws Exception {
    LOG.debug("<<< clearAllTable()");
    clearAllTable();
  }

  public void testCreateAndClean() throws Exception {
    CommonDataTest.createAllTable();
    CommonDataTest.clearAllTable();
  }

  public static void clearAllTable() throws Exception {
    clearUserTable();
    clearDeptTable();
    clearDictItemTable();
    clearDictTable();
  }

  public static void createAllTable() {
    createDictAndItem();
    createDept();
    createUser();
  }

  private static void createDept() {
    // Correct department dependencies
    SysDept deptA = new SysDept("A", "deptA");
    SysDept deptAA = new SysDept("AA", "deptAA");
    deptAA.setParentCode("A");
    SysDept deptAB = new SysDept("AB", "deptAB");
    deptAB.setParentCode("A");
    SysDept deptAAA = new SysDept("AAA", "deptAAA");
    deptAAA.setParentCode("AA");
    SysDept deptABA = new SysDept("ABA", "deptABA");
    deptABA.setParentCode("AB");

    List<SysDept> depts = new ArrayList<>();
    depts.addAll(Arrays.asList(deptA, deptAA, deptAB, deptAAA, deptABA));

    for (SysDept dept : depts) {
      Response response = deptRestApi.add(dept);
      assertDeptResponseSuccess(response);
    }

    JsonResponse<ListResult<SysDeptTree>> response = queryDeptTreeList();
    assertEquals(response.getAttributes().size(), 0);
    assertEquals(response.getResult().getTotal(), 5);
  }

  private static void createDictAndItem() {
    SysDict sexDict = new SysDict();
    sexDict.setDictCode("SYS_USER_SEX");
    sexDict.setDictName("name");
    sexDict.setDescription("description");
    Response response = dictRestApi.add(sexDict);
    assertDictResponseSuccess(response);

    SysDictItem sysDictItem = new SysDictItem();
    sysDictItem.setDictCode("SYS_USER_SEX");
    sysDictItem.setItemCode("SYS_USER_SEX_MALE");
    sysDictItem.setItemName("name");
    sysDictItem.setDescription("description");
    response = dictItemRestApi.add(sysDictItem);
    assertDictItemResponseSuccess(response);

    sysDictItem.setItemCode("SYS_USER_SEX_FEMALE");
    response = dictItemRestApi.add(sysDictItem);
    assertDictItemResponseSuccess(response);

    SysDict statusDict = new SysDict();
    statusDict.setDictCode("SYS_USER_STATUS");
    statusDict.setDictName("name");
    statusDict.setDescription("description");
    response = dictRestApi.add(statusDict);
    assertDictResponseSuccess(response);

    SysDictItem sysDictItem2 = new SysDictItem();
    sysDictItem2.setDictCode("SYS_USER_STATUS");
    sysDictItem2.setItemCode("SYS_USER_STATUS_AVAILABLE");
    sysDictItem2.setItemName("name");
    sysDictItem2.setDescription("description");
    response = dictItemRestApi.add(sysDictItem2);
    assertDictItemResponseSuccess(response);

    sysDictItem2.setItemCode("SYS_USER_STATUS_LOCKED");
    response = dictItemRestApi.add(sysDictItem2);
    assertDictItemResponseSuccess(response);

    sysDictItem2.setItemCode("SYS_USER_STATUS_REGISTERED");
    response = dictItemRestApi.add(sysDictItem2);
    assertDictItemResponseSuccess(response);

  }

  private static void createUser() {
    SysUser sysUser = new SysUser();
    sysUser.setUserName("user_name");
    sysUser.setRealName("real_name");
    sysUser.setPassword("password");
    sysUser.setAvatar("avatar");
    sysUser.setDeleted(1);
    sysUser.setPhone("123456789");
    sysUser.setRoleCode("roleCode");
    sysUser.setSex("SYS_USER_SEX_MALE");
    sysUser.setStatus("SYS_USER_STATUS_REGISTERED");
    sysUser.setEmail("test@submarine.org");
    sysUser.setBirthday(new Date());
    sysUser.setDeptCode("A");
    sysUser.setCreateTime(new Date());
    sysUser.setUpdateTime(new Date());

    Response response  = userRestApi.add(sysUser);
    JsonResponse<SysUser> jsonResponse = assertUserResponseSuccess(response);
    userId = jsonResponse.getResult().getId();
  }

  public static void clearUserTable() throws Exception {
    List<SysUser> userList = userService.queryPageList("", null, null, null, null, 0, 10);
    for (SysUser sysUser : userList) {
      userRestApi.delete(sysUser.getId());
    }
  }

  public static void clearDictItemTable() {
    Response response  = dictItemRestApi.list(null, null, null, null, null, null, 1, 10);
    assertDictItemResponseSuccess(response);
    JsonResponse<ListResult<SysDictItem>> jsonResponse = assertDictItemResponseSuccess(response);
    for (SysDictItem dictItem : jsonResponse.getResult().getRecords()) {
      dictItemRestApi.remove(dictItem.getId());
    }
  }

  public static void clearDictTable() {
    Response response  = dictRestApi.list(null, null, null, null, null, 1, 10);
    assertDictResponseSuccess(response);
    JsonResponse<ListResult<SysDict>> jsonResponse = assertDictResponseSuccess(response);
    for (SysDict dict : jsonResponse.getResult().getRecords()) {
      dictRestApi.remove(dict.getId());
    }
  }

  public static void clearDeptTable() {
    // clean department depends
    Response response = deptRestApi.resetParentDept();
    assertDeptResponseSuccess(response);

    // remove all test record
    response = deptRestApi.tree(null, null);
    JsonResponse<ListResult<SysDeptTree>> jsonResponse = wrapDeptResponse(response);
    assertTrue(jsonResponse.getSuccess());
    for (SysDeptTree deptTree : jsonResponse.getResult().getRecords()) {
      response = deptRestApi.remove(deptTree.getId());
      assertDeptResponseSuccess(response);
    }
  }

  public static JsonResponse<ListResult<SysDeptTree>> queryDeptTreeList() {
    Response response = deptRestApi.tree(null, null);
    JsonResponse<ListResult<SysDeptTree>> jsonResponse = wrapDeptResponse(response);

    assertTrue(jsonResponse.getSuccess());
    return jsonResponse;
  }

  public static JsonResponse<ListResult<SysDeptTree>> wrapDeptResponse(Response response) {
    String entity = (String) response.getEntity();
    Type type = new TypeToken<JsonResponse<ListResult<SysDeptTree>>>() {}.getType();
    JsonResponse<ListResult<SysDeptTree>> jsonResponse = gson.fromJson(entity, type);

    return jsonResponse;
  }

  public static JsonResponse<SysDept> assertDeptResponseSuccess(Response response) {
    String entity = (String) response.getEntity();
    Type type = new TypeToken<JsonResponse<SysDept>>() {}.getType();
    JsonResponse<SysDept> jsonResponse = gson.fromJson(entity, type);
    assertTrue(jsonResponse.getSuccess());
    return jsonResponse;
  }

  public static JsonResponse<SysUser> assertUserResponseSuccess(Response response) {
    String entity = (String) response.getEntity();
    Type type = new TypeToken<JsonResponse<SysUser>>() {}.getType();
    JsonResponse<SysUser> jsonResponse = gson.fromJson(entity, type);
    assertTrue(jsonResponse.getSuccess());
    return jsonResponse;
  }

  public static void assertResponseSuccess(Response response) {
    String entity = (String) response.getEntity();
    Type type = new TypeToken<JsonResponse>() {}.getType();
    JsonResponse jsonResponse = gson.fromJson(entity, type);
    Assert.assertTrue(jsonResponse.getSuccess());
  }

  public static JsonResponse<ListResult<SysDict>> assertDictResponseSuccess(Response response) {
    String entity = (String) response.getEntity();
    Type type = new TypeToken<JsonResponse<ListResult<SysDict>>>() {}.getType();
    JsonResponse<ListResult<SysDict>> jsonResponse = gson.fromJson(entity, type);
    assertTrue(jsonResponse.getSuccess());
    return jsonResponse;
  }

  public static JsonResponse<ListResult<SysDictItem>> assertDictItemResponseSuccess(Response response) {
    String entity = (String) response.getEntity();
    Type type = new TypeToken<JsonResponse<ListResult<SysDictItem>>>() {}.getType();
    JsonResponse<ListResult<SysDictItem>> jsonResponse = gson.fromJson(entity, type);
    assertTrue(jsonResponse.getSuccess());
    return jsonResponse;
  }
}
