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
package org.apache.submarine.server.workbench.rest;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.reflect.TypeToken;
import org.apache.submarine.server.workbench.database.entity.SysDict;
import org.apache.submarine.server.response.JsonResponse;
import org.apache.submarine.server.response.JsonResponse.ListResult;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;

import javax.ws.rs.core.Response;
import java.lang.reflect.Type;

import static junit.framework.TestCase.assertEquals;
import static junit.framework.TestCase.assertFalse;
import static junit.framework.TestCase.assertTrue;

public class SysDictRestApiTest {
  private static SysDictRestApi sysDictRestApi = new SysDictRestApi();
  private static SystemRestApi systemRestApi = new SystemRestApi();

  private static GsonBuilder gsonBuilder = new GsonBuilder();
  private static Gson gson = gsonBuilder.setDateFormat("yyyy-MM-dd HH:mm:ss").create();

  private static final int NEW_SYS_DICT_COUNT = 3;

  @BeforeClass
  public static void init() {
    for (int i = 0; i < NEW_SYS_DICT_COUNT; i++) {
      SysDict sysDict = new SysDict();
      sysDict.setDictCode("dictCode-SysDictRestApiTest-" + i);
      sysDict.setDictName("dictName-SysDictRestApiTest-" + i);
      sysDict.setDescription("desc-SysDictRestApiTest-" + i);
      Response response = sysDictRestApi.add(sysDict);
      CommonDataTest.assertResponseSuccess(response);
    }
  }

  @AfterClass
  public static void exit() {
    ListResult<SysDict> listResult = queryTestDictList();
    assertTrue(listResult.getRecords().size() > 0);
    assertTrue(listResult.getTotal() > 0);
    for (SysDict sysDict : listResult.getRecords()) {
      Response response = sysDictRestApi.remove(sysDict.getId());
      CommonDataTest.assertResponseSuccess(response);
    }

    //recheck
    ListResult listResult2 = queryTestDictList();
    assertEquals(listResult2.getTotal(), 0);
    assertEquals(listResult2.getRecords().size(), 0);
  }

  @Test
  public void hasDuplicateCheckTest() {
    Response response = systemRestApi.duplicateCheck(
        "sys_dict", "dict_code", "dictCode-SysDictRestApiTest-0", null, null, null);
    String entity = (String) response.getEntity();
    JsonResponse jsonResponse = gson.fromJson(entity, JsonResponse.class);
    assertFalse(jsonResponse.getSuccess());
  }

  @Test
  public void notDuplicateCheckTest() {
    Response response = systemRestApi.duplicateCheck("sys_dict", "dict_code", "not-exist-code",
        null, null, null);
    String entity = (String) response.getEntity();
    JsonResponse jsonResponse = gson.fromJson(entity, JsonResponse.class);
    assertTrue(jsonResponse.getSuccess());
  }

  @Test
  public void queryDictListTest() {
    ListResult<SysDict> listResult = queryTestDictList();
    assertEquals(listResult.getTotal(), NEW_SYS_DICT_COUNT);
    assertEquals(listResult.getRecords().size(), NEW_SYS_DICT_COUNT);
    assertTrue(listResult.getRecords().get(0) instanceof SysDict);
  }

  @Test
  public void setDeletedTest() {
    ListResult<SysDict> listResult = queryTestDictList();
    for (SysDict dict : listResult.getRecords()) {
      Response response = sysDictRestApi.delete(dict.getId(), 1);
      CommonDataTest.assertResponseSuccess(response);
    }

    ListResult<SysDict> listResult2 = queryTestDictList();
    for (SysDict dict : listResult2.getRecords()) {
      assertEquals((int) dict.getDeleted(), 1);
    }

    for (SysDict dict : listResult2.getRecords()) {
      Response response = sysDictRestApi.delete(dict.getId(), 0);
      CommonDataTest.assertResponseSuccess(response);
    }

    ListResult<SysDict> listResult3 = queryTestDictList();
    for (SysDict dict : listResult3.getRecords()) {
      assertEquals((int) dict.getDeleted(), 0);
    }
  }

  public static ListResult<SysDict> queryTestDictList() {
    Response response = sysDictRestApi.list("-SysDictRestApiTest-", "", "", "", "", 1, 10);
    String entity = (String) response.getEntity();
    Type type = new TypeToken<JsonResponse<ListResult<SysDict>>>() {}.getType();
    JsonResponse<ListResult<SysDict>> jsonResponse = gson.fromJson(entity, type);

    ListResult<SysDict> listResult = jsonResponse.getResult();
    return listResult;
  }
}
