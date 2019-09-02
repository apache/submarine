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
package org.apache.submarine.server;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.reflect.TypeToken;
import org.apache.submarine.database.entity.QueryResult;
import org.apache.submarine.database.entity.SysDict;
import org.junit.Test;

import javax.ws.rs.core.Response;
import java.lang.reflect.Type;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static junit.framework.TestCase.assertEquals;
import static junit.framework.TestCase.assertTrue;

public class JsonResponseTest {
  private GsonBuilder gsonBuilder = new GsonBuilder();
  private Gson gson = gsonBuilder.setDateFormat("yyyy-MM-dd HH:mm:ss").create();

  @Test
  public void serializObject() {
    SysDict sysDict = new SysDict();
    sysDict.setDictCode("code");
    sysDict.setDictName("name");
    sysDict.setDescription("desc");

    QueryResult<SysDict> queryResult = new QueryResult(Arrays.asList(sysDict), 1);
    Response response = new JsonResponse.Builder<QueryResult<SysDict>>(Response.Status.OK)
        .success(true).result(queryResult).build();

    String entity = (String) response.getEntity();

    Type type = new TypeToken<JsonResponse<QueryResult<SysDict>>>(){}.getType();

    JsonResponse<QueryResult<SysDict>> jsonResponse = gson.fromJson(entity, type);

    SysDict checkDict = jsonResponse.getResult().getRecords().get(0);
    assertEquals(checkDict.getDictCode(), "code");
    assertEquals(checkDict.getDictName(), "name");
    assertEquals(checkDict.getDescription(), "desc");
  }

  @Test
  public void serializQueryResult() {
    SysDict sysDict = new SysDict();
    sysDict.setDictCode("code");
    sysDict.setDictName("name");
    sysDict.setDescription("desc");

    List<SysDict> list = new ArrayList();
    list.add(sysDict);

    QueryResult<SysDict> queryResult = new QueryResult(list, 1);

    Response response = new JsonResponse.Builder<QueryResult<SysDict>>(Response.Status.OK)
        .success(true).result(queryResult).build();

    String entity = (String) response.getEntity();

    Type type = new TypeToken<JsonResponse<QueryResult<SysDict>>>(){}.getType();

    JsonResponse<QueryResult<SysDict>> jsonResponse = gson.fromJson(entity, type);

    QueryResult check = jsonResponse.getResult();
    assertTrue(check.getRecords().get(0) instanceof SysDict);
    assertEquals(check.getRecords().size(), queryResult.getRecords().size());
    assertEquals(check.getTotal(), queryResult.getTotal());
  }
}
