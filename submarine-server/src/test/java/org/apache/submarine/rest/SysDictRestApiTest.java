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
import org.apache.submarine.server.JsonResponse;
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
  private static final Gson gson = new Gson();
  private static final int NEW_SYS_DICT_COUNT = 3;

  @BeforeClass
  public static void init() {
    for (int i = 0; i < NEW_SYS_DICT_COUNT; i++) {
      SysDict sysDict = new SysDict();
      sysDict.setDictCode("dictCode-SysDictRestApiTest-" + i);
      sysDict.setDictName("dictName-SysDictRestApiTest-" + i);
      sysDict.setDescription("desc-SysDictRestApiTest-" + i);
      sysDictRestApi.add(sysDict);
    }
  }

  @AfterClass
  public static void exit() {
    QueryResult<SysDict> queryResult = queryTestDictList();
    assertTrue(queryResult.getRecords().size() > 0);
    assertTrue(queryResult.getTotal() > 0);
    for (SysDict sysDict : queryResult.getRecords()) {
      sysDictRestApi.remove(sysDict.getId());
    }

    //recheck
    QueryResult queryResult2 = queryTestDictList();
    assertEquals(queryResult2.getTotal(), 0);
    assertEquals(queryResult2.getRecords().size(), 0);
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
    QueryResult<SysDict> queryResult = queryTestDictList();
    assertEquals(queryResult.getTotal(), NEW_SYS_DICT_COUNT);
    assertEquals(queryResult.getRecords().size(), NEW_SYS_DICT_COUNT);
    assertTrue(queryResult.getRecords().get(0) instanceof SysDict);
  }

  @Test
  public void setDeletedTest() {
    QueryResult<SysDict> queryResult = queryTestDictList();
    for (SysDict dict : queryResult.getRecords()) {
      sysDictRestApi.delete(dict.getId(), 1);
    }

    QueryResult<SysDict> queryResult2 = queryTestDictList();
    for (SysDict dict : queryResult2.getRecords()) {
      assertEquals((int) dict.getDeleted(), 1);
    }

    for (SysDict dict : queryResult2.getRecords()) {
      sysDictRestApi.delete(dict.getId(), 0);
    }

    QueryResult<SysDict> queryResult3 = queryTestDictList();
    for (SysDict dict : queryResult3.getRecords()) {
      assertEquals((int) dict.getDeleted(), 0);
    }
  }

  public static QueryResult<SysDict> queryTestDictList() {
    Response response = sysDictRestApi.list("-SysDictRestApiTest-", "", "", "", "", 1, 10);
    String entity = (String) response.getEntity();
    Type type = new TypeToken<JsonResponse<QueryResult<SysDict>>>(){}.getType();
    JsonResponse<QueryResult<SysDict>> jsonResponse = gson.fromJson(entity, type);

    QueryResult<SysDict> queryResult = jsonResponse.getResult();
    return queryResult;
  }
}
