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
import org.apache.submarine.server.model.database.entities.RegisteredModelEntity;
import org.apache.submarine.server.model.database.service.RegisteredModelService;
import org.apache.submarine.server.utils.gson.ExperimentIdDeserializer;
import org.apache.submarine.server.utils.gson.ExperimentIdSerializer;

public class RegisteredModelRestApiTest {
  private final RegisteredModelService registeredModelService = new RegisteredModelService();
  private final String registeredModelName = "testRegisteredModel";
  private final String newRegisteredModelName = "newTestRegisteredModel";
  private final String registeredModelDescription = "test registered model description";
  private final String newRegisteredModelDescription = "new test registered model description";
  private final String registeredModelTag = "testTag";
  private final String defaultRegisteredModelTag = "defaultTag";
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
    List<String> tags = new ArrayList<>();
    tags.add(defaultRegisteredModelTag);
    registeredModel.setTags(tags);
    registeredModelService.insert(registeredModel);
  }

  @Test
  public void testListRegisteredModel(){
    Response listRegisteredModelResponse = registeredModelRestApi.listRegisteredModels();
    List<RegisteredModelEntity> result = getResultListFromResponse(
        listRegisteredModelResponse, RegisteredModelEntity.class);
    assertEquals(1, result.size());
    verifyResult(registeredModel, result.get(0));
  }

  @Test
  public void testGetModelRegisteredModel(){
    Response getRegisteredModelResponse = registeredModelRestApi.getRegisteredModel(registeredModelName);
    RegisteredModelEntity result = getResultFromResponse(
        getRegisteredModelResponse, RegisteredModelEntity.class);
    verifyResult(registeredModel, result);
  }

  @Test
  public void testAddAndDeleteRegisteredModelTag(){
    registeredModelRestApi.deleteRegisteredModelTag(registeredModelName, defaultRegisteredModelTag);
    Response getRegisteredModelResponse = registeredModelRestApi.getRegisteredModel(registeredModelName);
    RegisteredModelEntity result = getResultFromResponse(
        getRegisteredModelResponse , RegisteredModelEntity.class);
    assertEquals(0, result.getTags().size());

    registeredModelRestApi.createRegisteredModelTag(registeredModelName, registeredModelTag);
    getRegisteredModelResponse = registeredModelRestApi.getRegisteredModel(registeredModelName);
    result = getResultFromResponse(
        getRegisteredModelResponse, RegisteredModelEntity.class);
    assertEquals(1, result.getTags().size());
    assertEquals(registeredModelTag, result.getTags().get(0));


  }

  @Test
  public void testUpdateRegisteredModel(){
    RegisteredModelEntity newRegisteredModel = new RegisteredModelEntity();
    newRegisteredModel.setName(newRegisteredModelName);
    newRegisteredModel.setDescription(newRegisteredModelDescription);
    List<String> tags = new ArrayList<>();
    tags.add(defaultRegisteredModelTag);
    newRegisteredModel.setTags(tags);
    registeredModelRestApi.updateRegisteredModel(registeredModelName, newRegisteredModel);
    Response getRegisteredModelResponse = registeredModelRestApi.getRegisteredModel(newRegisteredModelName);
    RegisteredModelEntity result = getResultFromResponse(
        getRegisteredModelResponse , RegisteredModelEntity.class);
    verifyResult(newRegisteredModel, result);
  }

  @Test
  public void testDeleteRegisteredModel(){
    registeredModelRestApi.deleteRegisteredModel(registeredModelName);
    Response listRegisteredModelResponse = registeredModelRestApi.listRegisteredModels();
    List<RegisteredModelEntity> result = getResultListFromResponse(
        listRegisteredModelResponse, RegisteredModelEntity.class);
    assertEquals(0, result.size());
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
    for ( int i = 0; i < result.getTags().size(); i++ ){
      assertEquals(result.getTags().get(i), actual.getTags().get(i));
    }
  }
}
