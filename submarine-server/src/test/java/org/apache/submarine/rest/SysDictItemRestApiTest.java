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
import com.google.gson.reflect.TypeToken;
import org.apache.submarine.database.entity.QueryResult;
import org.apache.submarine.database.entity.SysDict;
import org.apache.submarine.database.entity.SysDictItem;
import org.apache.submarine.server.JsonResponse;
import org.junit.AfterClass;
import org.junit.Assert;
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
  private static final Gson gson = new Gson();
  private static final int NEW_SYS_DICT_ITEM_COUNT = 3;
  private static final String DICT_CODE = "dictCode-SysDictItemRestApiTest";

  @BeforeClass
  public static void init() {
    SysDict sysDict = new SysDict();
    sysDict.setDictCode(DICT_CODE);
    sysDict.setDictName("dictName-SysDictItemRestApiTest");
    sysDict.setDescription("description-SysDictItemRestApiTest");
    Response response = sysDictRestApi.add(sysDict);
    assertResponseSuccess(response);

    for (int i = 0; i < NEW_SYS_DICT_ITEM_COUNT; i++) {
      SysDictItem sysDictItem = new SysDictItem();
      sysDictItem.setDictCode(DICT_CODE);
      sysDictItem.setItemCode("text-SysDictItemRestApiTest-" + i);
      sysDictItem.setItemName("value-SysDictItemRestApiTest-" + i);
      sysDictItem.setDescription("desc-SysDictItemRestApiTest-" + i);
      Response response2 = sysDictItemRestApi.add(sysDictItem);
      assertResponseSuccess(response2);
    }
  }

  @AfterClass
  public static void exit() {
    QueryResult<SysDictItem> queryResult = queryTestDictItemList();
    for (SysDictItem sysDictItem : queryResult.getRecords()) {
      Response response = sysDictItemRestApi.remove(sysDictItem.getId());
      assertResponseSuccess(response);
    }
    //recheck
    QueryResult queryResult2 = queryTestDictItemList();
    assertEquals(queryResult2.getTotal(), 0);
    assertEquals(queryResult2.getRecords().size(), 0);

    //recheck
    QueryResult<SysDict> queryResult3 = queryTestDictList();
    assertTrue(queryResult3.getRecords().size() > 0);
    assertTrue(queryResult3.getTotal() > 0);
    for (SysDict sysDict : queryResult3.getRecords()) {
      Response response = sysDictRestApi.remove(sysDict.getId());
      assertResponseSuccess(response);
    }
    QueryResult<SysDict> queryResult4 = queryTestDictList();
    assertEquals(queryResult4.getTotal(), 0);
    assertEquals(queryResult4.getRecords().size(), 0);
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
        "sys_dict_item", "item_name", "not-exist-code", "dict_code", DICT_CODE, null);
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
    QueryResult<SysDictItem> queryResult = queryTestDictItemList();
    assertEquals(queryResult.getTotal(), NEW_SYS_DICT_ITEM_COUNT);
    assertEquals(queryResult.getRecords().size(), NEW_SYS_DICT_ITEM_COUNT);
    assertTrue(queryResult.getRecords().get(0) instanceof SysDictItem);
  }

  @Test
  public void setDeletedTest() {
    QueryResult<SysDictItem> queryResult = queryTestDictItemList();
    for (SysDictItem item : queryResult.getRecords()) {
      sysDictItemRestApi.delete(item.getId(), 1);
    }

    QueryResult<SysDictItem> queryResult2 = queryTestDictItemList();
    for (SysDictItem item : queryResult2.getRecords()) {
      assertEquals((int) item.getDeleted(), 1);
    }

    for (SysDictItem item : queryResult2.getRecords()) {
      sysDictItemRestApi.delete(item.getId(), 0);
    }

    QueryResult<SysDictItem> queryResult3 = queryTestDictItemList();
    for (SysDictItem item : queryResult3.getRecords()) {
      assertEquals((int) item.getDeleted(), 0);
    }
  }

  public static QueryResult<SysDictItem> queryTestDictItemList() {
    Response response = sysDictItemRestApi.list(DICT_CODE, "", "", "", "", "", 0, 10);
    String entity = (String) response.getEntity();
    Type type = new TypeToken<JsonResponse<QueryResult<SysDictItem>>>(){}.getType();
    JsonResponse<QueryResult<SysDictItem>> jsonResponse = gson.fromJson(entity, type);

    QueryResult<SysDictItem> queryResult = jsonResponse.getResult();
    return queryResult;
  }

  public static QueryResult<SysDict> queryTestDictList() {
    Response response = sysDictRestApi.list("SysDictItemRestApiTest", "", "", "", "", 1, 10);
    String entity = (String) response.getEntity();
    Type type = new TypeToken<JsonResponse<QueryResult<SysDict>>>(){}.getType();
    JsonResponse<QueryResult<SysDict>> jsonResponse = gson.fromJson(entity, type);

    QueryResult queryResult = jsonResponse.getResult();
    return queryResult;
  }

  private static JsonResponse<QueryResult<SysDict>> wrapResponse(Response response) {
    String entity = (String) response.getEntity();
    Type type = new TypeToken<JsonResponse<QueryResult<SysDict>>>() {}.getType();
    JsonResponse<QueryResult<SysDict>> jsonResponse = gson.fromJson(entity, type);

    return jsonResponse;
  }

  private static void assertResponseSuccess(Response response) {
    JsonResponse<QueryResult<SysDict>> jsonResponse = wrapResponse(response);
    Assert.assertTrue(jsonResponse.getSuccess());
  }
}
