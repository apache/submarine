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
import java.util.List;

import static junit.framework.TestCase.assertEquals;

public class JsonResponseTest {

  @Test
  public void serializObject() {
    GsonBuilder gsonBuilder = new GsonBuilder();
    Gson gson = gsonBuilder.create();

    SysDict sysDict = new SysDict();
    sysDict.setDictCode("code");
    sysDict.setDictName("name");
    sysDict.setDescription("desc");

    Response response = new JsonResponse.Builder<SysDict>(Response.Status.OK)
        .success(true).result(sysDict).build();

    String entity = (String) response.getEntity();

    Type type = new TypeToken<JsonResponse<SysDict>>(){}.getType();

    JsonResponse<SysDict> jsonResponse = gson.fromJson(entity, type);

    SysDict checkDict = jsonResponse.getResult();
    assertEquals(checkDict.getDictCode(), "code");
    assertEquals(checkDict.getDictName(), "name");
    assertEquals(checkDict.getDescription(), "desc");
  }

  @Test
  public void serializQueryResult() {
    GsonBuilder gsonBuilder = new GsonBuilder();
    Gson gson = gsonBuilder.create();

    SysDict sysDict = new SysDict();
    sysDict.setDictCode("code");
    sysDict.setDictName("name");
    sysDict.setDescription("desc");

    List<SysDict> list = new ArrayList();
    list.add(sysDict);

    QueryResult queryResult = new QueryResult(list, 1);

    Response response = new JsonResponse.Builder<QueryResult>(Response.Status.OK)
        .success(true).result(queryResult).build();

    String entity = (String) response.getEntity();

    Type type = new TypeToken<JsonResponse<QueryResult>>(){}.getType();

    JsonResponse<QueryResult> jsonResponse = gson.fromJson(entity, type);

    QueryResult check = jsonResponse.getResult();
    assertEquals(check.getRecords().size(), queryResult.getRecords().size());
    assertEquals(check.getTotal(), queryResult.getTotal());
  }
}
