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
package org.apache.submarine.server.rest.workbench;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.reflect.TypeToken;

import org.apache.submarine.server.utils.response.JsonResponse;
import org.apache.submarine.server.utils.response.JsonResponse.ListResult;
import org.apache.submarine.server.database.workbench.entity.SysDictEntity;
import org.apache.submarine.server.database.workbench.entity.SysDictItemEntity;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;

import javax.ws.rs.core.Response;
import java.lang.reflect.Type;

import static junit.framework.TestCase.assertEquals;
import static junit.framework.TestCase.assertFalse;
import static junit.framework.TestCase.assertTrue;

public class SysDictItemRestApiTest {
  private static SysDictRestApi sysDictRestApi = new SysDictRestApi();
  private static SysDictItemRestApi sysDictItemRestApi = new SysDictItemRestApi();
  private static SystemRestApi systemRestApi = new SystemRestApi();
  private static GsonBuilder gsonBuilder = new GsonBuilder();
  private static Gson gson = gsonBuilder.setDateFormat("yyyy-MM-dd HH:mm:ss").create();
  private static final int NEW_SYS_DICT_ITEM_COUNT = 3;
  private static final String DICT_CODE = "dictCode-SysDictItemRestApiTest";

  @BeforeClass
  public static void init() {
    SysDictEntity sysDict = new SysDictEntity();
    sysDict.setDictCode(DICT_CODE);
    sysDict.setDictName("dictName-SysDictItemRestApiTest");
    sysDict.setDescription("description-SysDictItemRestApiTest");
    Response response = sysDictRestApi.add(sysDict);
    CommonDataTest.assertResponseSuccess(response);

    for (int i = 0; i < NEW_SYS_DICT_ITEM_COUNT; i++) {
      SysDictItemEntity sysDictItem = new SysDictItemEntity();
      sysDictItem.setDictCode(DICT_CODE);
      sysDictItem.setItemCode("text-SysDictItemRestApiTest-" + i);
      sysDictItem.setItemName("value-SysDictItemRestApiTest-" + i);
      sysDictItem.setDescription("desc-SysDictItemRestApiTest-" + i);
      Response response2 = sysDictItemRestApi.add(sysDictItem);
      CommonDataTest.assertResponseSuccess(response2);
    }
  }

  @AfterClass
  public static void exit() {
    ListResult<SysDictItemEntity> listResult = queryTestDictItemList();
    for (SysDictItemEntity sysDictItem : listResult.getRecords()) {
      Response response = sysDictItemRestApi.remove(sysDictItem.getId());
      CommonDataTest.assertResponseSuccess(response);
    }
    //recheck
    ListResult listResult2 = queryTestDictItemList();
    assertEquals(listResult2.getTotal(), 0);
    assertEquals(listResult2.getRecords().size(), 0);

    //recheck
    ListResult<SysDictEntity> listResult3 = queryTestDictList();
    assertTrue(listResult3.getRecords().size() == 1);
    assertTrue(listResult3.getTotal() == 1);
    for (SysDictEntity sysDict : listResult3.getRecords()) {
      Response response = sysDictRestApi.remove(sysDict.getId());
      CommonDataTest.assertResponseSuccess(response);
    }
    ListResult<SysDictEntity> listResult4 = queryTestDictList();
    assertEquals(listResult4.getTotal(), 0);
    assertEquals(listResult4.getRecords().size(), 0);
  }

  @Test
  public void hasDuplicateCheckItemCodeTest() {
    Response response = systemRestApi.duplicateCheck(
        "sys_dict_item", "item_code", "text-SysDictItemRestApiTest-0", "dict_code", DICT_CODE, null);
    String entity = (String) response.getEntity();
    JsonResponse jsonResponse = gson.fromJson(entity, JsonResponse.class);
    assertFalse(jsonResponse.getSuccess());
  }

  @Test
  public void hasDuplicateCheckItemNameTest() {
    Response response = systemRestApi.duplicateCheck(
        "sys_dict_item", "item_name", "value-SysDictItemRestApiTest-0", "dict_code", DICT_CODE, null);
    String entity = (String) response.getEntity();
    JsonResponse jsonResponse = gson.fromJson(entity, JsonResponse.class);
    assertFalse(jsonResponse.getSuccess());
  }

  @Test
  public void notDuplicateCheckItemCodeTest() {
    Response response = systemRestApi.duplicateCheck(
        "sys_dict_item", "item_code", "not-exist-code", "dict_code", DICT_CODE, null);
    String entity = (String) response.getEntity();
    JsonResponse jsonResponse = gson.fromJson(entity, JsonResponse.class);
    assertTrue(jsonResponse.getSuccess());
  }

  @Test
  public void notDuplicateCheckItemNameTest() {
    Response response = systemRestApi.duplicateCheck(
        "sys_dict_item", "item_name", "not-exist-code", "dict_code", DICT_CODE, null);
    String entity = (String) response.getEntity();
    JsonResponse jsonResponse = gson.fromJson(entity, JsonResponse.class);
    assertTrue(jsonResponse.getSuccess());
  }

  @Test
  public void queryDictItemListTest() {
    ListResult<SysDictItemEntity> listResult = queryTestDictItemList();
    assertEquals(listResult.getTotal(), NEW_SYS_DICT_ITEM_COUNT);
    assertEquals(listResult.getRecords().size(), NEW_SYS_DICT_ITEM_COUNT);
    assertTrue(listResult.getRecords().get(0) instanceof SysDictItemEntity);
  }

  @Test
  public void setDeletedTest() {
    ListResult<SysDictItemEntity> listResult = queryTestDictItemList();
    for (SysDictItemEntity item : listResult.getRecords()) {
      sysDictItemRestApi.delete(item.getId(), 1);
    }

    ListResult<SysDictItemEntity> listResult2 = queryTestDictItemList();
    for (SysDictItemEntity item : listResult2.getRecords()) {
      assertEquals((int) item.getDeleted(), 1);
    }

    for (SysDictItemEntity item : listResult2.getRecords()) {
      sysDictItemRestApi.delete(item.getId(), 0);
    }

    ListResult<SysDictItemEntity> listResult3 = queryTestDictItemList();
    for (SysDictItemEntity item : listResult3.getRecords()) {
      assertEquals((int) item.getDeleted(), 0);
    }
  }

  public static ListResult<SysDictItemEntity> queryTestDictItemList() {
    Response response = sysDictItemRestApi.list(DICT_CODE, "", "", "", "", "", 0, 10);
    String entity = (String) response.getEntity();
    Type type = new TypeToken<JsonResponse<ListResult<SysDictItemEntity>>>() {}.getType();
    JsonResponse<ListResult<SysDictItemEntity>> jsonResponse = gson.fromJson(entity, type);

    ListResult<SysDictItemEntity> listResult = jsonResponse.getResult();
    return listResult;
  }

  public static ListResult<SysDictEntity> queryTestDictList() {
    Response response = sysDictRestApi.list(DICT_CODE, "", "", "", "", 1, 10);
    String entity = (String) response.getEntity();
    Type type = new TypeToken<JsonResponse<ListResult<SysDictEntity>>>() {}.getType();
    JsonResponse<ListResult<SysDictEntity>> jsonResponse = gson.fromJson(entity, type);

    ListResult listResult = jsonResponse.getResult();
    return listResult;
  }
}
