/**
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * <p>
 * http://www.apache.org/licenses/LICENSE-2.0
 * <p>
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
import org.apache.submarine.database.entity.SysDict;
import org.apache.submarine.server.JsonResponse.ListResult;
import org.junit.Test;

import javax.ws.rs.core.Response;
import java.lang.reflect.Type;
import java.util.ArrayList;
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

    Response response = new JsonResponse.Builder<SysDict>(Response.Status.OK)
        .success(true).result(sysDict).build();

    String entity = (String) response.getEntity();

    Type type = new TypeToken<JsonResponse<SysDict>>() {
    }.getType();

    JsonResponse<SysDict> jsonResponse = gson.fromJson(entity, type);

    SysDict checkDict = jsonResponse.getResult();
    assertEquals(checkDict.getDictCode(), "code");
    assertEquals(checkDict.getDictName(), "name");
    assertEquals(checkDict.getDescription(), "desc");
  }

  @Test
  public void serializListResult() {
    SysDict sysDict = new SysDict();
    sysDict.setDictCode("code");
    sysDict.setDictName("name");
    sysDict.setDescription("desc");

    List<SysDict> list = new ArrayList();
    list.add(sysDict);

    ListResult<SysDict> listResult = new ListResult(list, list.size());

    Response response = new JsonResponse.Builder<ListResult<SysDict>>(Response.Status.OK)
        .success(true).result(listResult).build();

    String entity = (String) response.getEntity();

    Type type = new TypeToken<JsonResponse<ListResult<SysDict>>>() {
    }.getType();

    JsonResponse<ListResult<SysDict>> jsonResponse = gson.fromJson(entity, type);

    ListResult<SysDict> check = jsonResponse.getResult();
    assertTrue(check.getRecords().get(0) instanceof SysDict);
    assertEquals(check.getRecords().size(), listResult.getRecords().size());
    assertEquals(check.getTotal(), listResult.getTotal());
    assertEquals(check.getRecords().get(0).getDictCode(), sysDict.getDictCode());
    assertEquals(check.getRecords().get(0).getDictName(), sysDict.getDictName());
    assertEquals(check.getRecords().get(0).getDescription(), sysDict.getDescription());
  }
}
