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

package org.apache.submarine.server.rest;

import static org.junit.Assert.assertEquals;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import java.util.ArrayList;
import java.util.List;
import javax.ws.rs.core.Response;
import org.apache.submarine.server.api.experiment.ExperimentId;
import org.apache.submarine.server.api.spec.RegisteredModelTagSpec;
import org.apache.submarine.server.gson.ExperimentIdDeserializer;
import org.apache.submarine.server.gson.ExperimentIdSerializer;
import org.apache.submarine.server.model.database.entities.RegisteredModelEntity;
import org.apache.submarine.server.model.database.service.RegisteredModelService;

public class RegisteredModelRestApiTest {
  private final RegisteredModelService registeredModelService = new RegisteredModelService();
  private final String registeredModelName = "test_registered_model";
  private final String registeredModelDescription = "test registered model description";
  private final String registeredModelTag = "testTag";
  private static final GsonBuilder gsonBuilder = new GsonBuilder()
      .registerTypeAdapter(ExperimentId.class, new ExperimentIdSerializer())
      .registerTypeAdapter(ExperimentId.class, new ExperimentIdDeserializer());
  private static Gson gson = gsonBuilder.setDateFormat("yyyy-MM-dd HH:mm:ss").create();
  private RegisteredModelEntity registeredModel = new RegisteredModelEntity();

  private final RegisteredModelRestApi registeredModelRestApi = new RegisteredModelRestApi();


  @Before
  public void testCreateRegisteredModel() {
    registeredModel.setName(registeredModelName);
    registeredModel.setDescription(registeredModelDescription);
    registeredModelService.insert(registeredModel);
  }

  @Test
  public void testListRegisteredModel(){
    Response listRegisteredModelResponse = registeredModelRestApi.listRegisteredModels();
    List<RegisteredModelEntity> result = getResultListFromResponse(
        listRegisteredModelResponse, RegisteredModelEntity.class);
    assertEquals(result.size(), 1);
    verifyResult(result.get(0), registeredModel);
  }

  @Test
  public void testGetModelRegisteredModel(){
    Response getRegisteredModelResponse = registeredModelRestApi.getRegisteredModel(registeredModelName);
    RegisteredModelEntity result = getResultFromResponse(
        getRegisteredModelResponse, RegisteredModelEntity.class);
    verifyResult(result, registeredModel);
  }

  @Test
  public void testAddRegisteredModelTag(){
    RegisteredModelTagSpec spec = new RegisteredModelTagSpec();
    spec.setName(registeredModelName);
    spec.setTag(registeredModelTag);
    registeredModelRestApi.createRegisteredModelTag(spec);
    Response getRegisteredModelResponse = registeredModelRestApi.getRegisteredModel(registeredModelName);
    RegisteredModelEntity result = getResultFromResponse(
        getRegisteredModelResponse, RegisteredModelEntity.class);
    assertEquals(result.getTags().size(), 1);
    assertEquals(result.getTags().get(0), registeredModelTag);
  }

  @Test
  public void testDeleteRegisteredModelTag(){
    RegisteredModelTagSpec spec = new RegisteredModelTagSpec();
    spec.setName(registeredModelName);
    spec.setTag(registeredModelTag);
    registeredModelRestApi.deleteRegisteredModelTag(spec);
    Response getRegisteredModelResponse = registeredModelRestApi.getRegisteredModel(registeredModelName);
    RegisteredModelEntity result = getResultFromResponse(
        getRegisteredModelResponse , RegisteredModelEntity.class);
    assertEquals(result.getTags().size(), 0);
  }

  @Test
  public void testDeleteRegisteredModel(){
    registeredModelRestApi.deleteRegisteredModel(registeredModelName);
    Response listRegisteredModelResponse = registeredModelRestApi.listRegisteredModels();
    List<RegisteredModelEntity> result = getResultListFromResponse(
        listRegisteredModelResponse, RegisteredModelEntity.class);
    assertEquals(result.size(), 0);
  }

  @After
  public void tearDown(){
    registeredModelService.deleteAll();
  }

  private <T> T getResultFromResponse(Response response, Class<T> typeT) {
    String entity = (String) response.getEntity();
    JsonObject object = new JsonParser().parse(entity).getAsJsonObject();
    JsonElement result = object.get("result");
    return gson.fromJson(result, typeT);
  }

  private <T> List<T> getResultListFromResponse(Response response, Class<T> typeT) {
    String entity = (String) response.getEntity();
    JsonObject object = new JsonParser().parse(entity).getAsJsonObject();
    JsonElement result = object.get("result");
    List<T> list = new ArrayList<T>();
    JsonArray array = result.getAsJsonArray();
    for (JsonElement jsonElement : array) {
      list.add(gson.fromJson(jsonElement, typeT));
    }
    return list;
  }

  private void verifyResult(RegisteredModelEntity result, RegisteredModelEntity actual){
    assertEquals(result.getName(), actual.getName());
    assertEquals(result.getDescription(), actual.getDescription());
  }
}
