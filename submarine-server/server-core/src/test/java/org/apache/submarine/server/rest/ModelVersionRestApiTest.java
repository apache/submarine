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
import org.junit.Ignore;
import org.junit.Test;
import java.util.ArrayList;
import java.util.List;
import javax.ws.rs.core.Response;
import org.apache.submarine.server.api.experiment.ExperimentId;
import org.apache.submarine.server.model.database.entities.ModelVersionEntity;
import org.apache.submarine.server.model.database.entities.RegisteredModelEntity;
import org.apache.submarine.server.model.database.service.ModelVersionService;
import org.apache.submarine.server.model.database.service.RegisteredModelService;
import org.apache.submarine.server.utils.ExperimentIdDeserializer;
import org.apache.submarine.server.utils.ExperimentIdSerializer;


public class ModelVersionRestApiTest {
  private ModelVersionRestApi modelVersionRestApi = new ModelVersionRestApi();
  private final String registeredModelName = "testRegisteredModel";
  private final String registeredModelDescription = "test registered model description";
  private final String modelVersionDescription = "test model version description";
  private final String newModelVersionDescription = "new test registered model description";
  private final String modelVersionId = "model_version_id";
  private final String modelVersionId2 = "model_version_id2";
  private final String modelVersionUserId = "test123";
  private final String modelVersionExperimentId = "experiment_123";
  private final String modelVersionModelType = "experiment_123";
  private final String modelVersionTag = "testTag";

  private final RegisteredModelService registeredModelService = new RegisteredModelService();

  private final ModelVersionService modelVersionService = new ModelVersionService();

  private static final GsonBuilder gsonBuilder = new GsonBuilder()
      .registerTypeAdapter(ExperimentId.class, new ExperimentIdSerializer())
      .registerTypeAdapter(ExperimentId.class, new ExperimentIdDeserializer());
  private static Gson gson = gsonBuilder.setDateFormat("yyyy-MM-dd HH:mm:ss").create();

  private ModelVersionEntity modelVersion1 = new ModelVersionEntity();

  private ModelVersionEntity modelVersion2 = new ModelVersionEntity();


  @Before
  public void createModelVersion() {
    RegisteredModelEntity registeredModel = new RegisteredModelEntity();
    registeredModel.setName(registeredModelName);
    registeredModel.setDescription(registeredModelDescription);
    registeredModelService.insert(registeredModel);
    modelVersion1.setName(registeredModelName);
    modelVersion1.setDescription(String.format("%s1", modelVersionDescription));
    modelVersion1.setVersion(1);
    modelVersion1.setId(modelVersionId);
    modelVersion1.setUserId(modelVersionUserId);
    modelVersion1.setExperimentId(modelVersionExperimentId);
    modelVersion1.setModelType(modelVersionModelType);
    modelVersionService.insert(modelVersion1);

    modelVersion2.setName(registeredModelName);
    modelVersion2.setDescription(String.format("%s2", modelVersionDescription));
    modelVersion2.setVersion(2);
    modelVersion2.setId(modelVersionId2);
    modelVersion2.setUserId(modelVersionUserId);
    modelVersion2.setExperimentId(modelVersionExperimentId);
    modelVersion2.setModelType(modelVersionModelType);
    modelVersionService.insert(modelVersion2);
  }

  @After
  public void tearDown(){
    registeredModelService.deleteAll();
  }

  @Test
  public void testListModelVersion(){
    Response listModelVersionResponse = modelVersionRestApi.listModelVersions(registeredModelName);
    List<ModelVersionEntity> result = getResultListFromResponse(
        listModelVersionResponse, ModelVersionEntity.class);
    assertEquals(2, result.size());
    verifyResult(modelVersion1, result.get(0));
    verifyResult(modelVersion2, result.get(1));
  }

  @Test
  public void testGetModelVersion(){
    Response getModelVersionResponse = modelVersionRestApi.getModelVersion(registeredModelName, 1);
    ModelVersionEntity result = getResultFromResponse(getModelVersionResponse, ModelVersionEntity.class);
    verifyResult(modelVersion1, result);
  }

  @Test
  public void testAddAndDeleteModelVersionTag(){
    modelVersionRestApi.createModelVersionTag(registeredModelName, "1", modelVersionTag);
    Response getModelVersionResponse = modelVersionRestApi.getModelVersion(registeredModelName, 1);
    ModelVersionEntity result = getResultFromResponse(
        getModelVersionResponse, ModelVersionEntity.class);
    assertEquals(1, result.getTags().size());
    assertEquals(modelVersionTag, result.getTags().get(0));

    modelVersionRestApi.deleteModelVersionTag(registeredModelName, "1", modelVersionTag);
    getModelVersionResponse = modelVersionRestApi.getModelVersion(registeredModelName, 1);
    result = getResultFromResponse(
        getModelVersionResponse , ModelVersionEntity.class);
    assertEquals(0, result.getTags().size());
  }

  @Test
  public void testUpdateModelVersion(){
    ModelVersionEntity newModelVersion = new ModelVersionEntity();
    newModelVersion.setName(registeredModelName);
    newModelVersion.setVersion(1);
    newModelVersion.setDescription(newModelVersionDescription);
    modelVersionRestApi.updateModelVersion(newModelVersion);
    Response getModelVersionResponse = modelVersionRestApi.getModelVersion(registeredModelName, 1);
    ModelVersionEntity result = getResultFromResponse(
        getModelVersionResponse , ModelVersionEntity.class);
    assertEquals(newModelVersionDescription, result.getDescription());
  }

  // TODO(KUAN-HSUN LI): mock the s3Client
  @Ignore
  @Test
  public void testDeleteModelVersion(){
    modelVersionRestApi.deleteModelVersion(registeredModelName, 1);
    Response listModelVersionResponse = modelVersionRestApi.listModelVersions(registeredModelName);
    List<ModelVersionEntity> result = getResultListFromResponse(
        listModelVersionResponse, ModelVersionEntity.class);
    assertEquals(1, result.size());
    verifyResult(modelVersion2, result.get(0));
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

  private void verifyResult(ModelVersionEntity result, ModelVersionEntity actual){
    assertEquals(result.getName(), actual.getName());
    assertEquals(result.getDescription(), actual.getDescription());
    assertEquals(result.getVersion(), actual.getVersion());
    assertEquals(result.getId(), actual.getId());
    assertEquals(result.getExperimentId(), actual.getExperimentId());
    assertEquals(result.getModelType(), actual.getModelType());
  }
}
