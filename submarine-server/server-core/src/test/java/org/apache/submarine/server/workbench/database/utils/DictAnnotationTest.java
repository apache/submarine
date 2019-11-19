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
package org.apache.submarine.server.workbench.database.utils;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.internal.LinkedTreeMap;
import com.google.gson.reflect.TypeToken;
import org.apache.submarine.server.workbench.rest.CommonDataTest;
import org.apache.submarine.server.workbench.rest.SysUserRestApi;
import org.apache.submarine.server.workbench.server.JsonResponse;
import org.junit.Test;

import javax.ws.rs.core.Response;

import java.lang.reflect.Type;
import java.util.ArrayList;

import static org.junit.Assert.assertTrue;

public class DictAnnotationTest extends CommonDataTest {
  private SysUserRestApi userRestApi = new SysUserRestApi();

  private static GsonBuilder gsonBuilder = new GsonBuilder();
  private static Gson gson = gsonBuilder.setDateFormat("yyyy-MM-dd HH:mm:ss").create();

  @Test
  public void userSexDictAnnotationTest() {
    Response response = userRestApi.queryPageList(null, null, null, null, null, 1, 10);

    String entity = (String) response.getEntity();
    Type type = new TypeToken<JsonResponse>() {}.getType();
    JsonResponse jsonResponse = gson.fromJson(entity, type);

    LinkedTreeMap<String, Object> linkedTreeMap = (LinkedTreeMap<String, Object>) jsonResponse.getResult();
    ArrayList<LinkedTreeMap<String, Object>> arrayList
        = (ArrayList<LinkedTreeMap<String, Object>>) linkedTreeMap.get("records");

    assertTrue(arrayList.get(0).containsKey("sex"));
    assertTrue(arrayList.get(0).containsKey("sex" + DictAnnotation.DICT_SUFFIX));

    assertTrue(arrayList.get(0).containsKey("status"));
    assertTrue(arrayList.get(0).containsKey("status" + DictAnnotation.DICT_SUFFIX));
  }
}
