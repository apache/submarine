/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.submarine.server;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.reflect.TypeToken;
import org.apache.submarine.database.entity.QueryResult;
import org.apache.submarine.database.entity.SysUser;
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

    SysUser sysUser = new SysUser();
    sysUser.setName("name");
    sysUser.setToken("token");
    sysUser.setEmail("email");

    Response response = new JsonResponse.Builder<SysUser>(Response.Status.OK)
        .success(true).result(sysUser).build();

    String entity = (String) response.getEntity();

    Type type = new TypeToken<JsonResponse<SysUser>>(){}.getType();

    JsonResponse<SysUser> jsonResponse = gson.fromJson(entity, type);

    SysUser check = jsonResponse.getResult();
    assertEquals(check.getName(), "name");
    assertEquals(check.getToken(), "token");
    assertEquals(check.getEmail(), "email");
  }

  @Test
  public void serializQueryResult() {
    GsonBuilder gsonBuilder = new GsonBuilder();
    Gson gson = gsonBuilder.create();

    SysUser sysUser = new SysUser();
    sysUser.setName("name");
    sysUser.setToken("token");
    sysUser.setEmail("email");

    List<SysUser> list = new ArrayList();
    list.add(sysUser);

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
