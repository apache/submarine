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
import org.apache.submarine.server.gson.ExperimentIdDeserializer;
import org.apache.submarine.server.gson.ExperimentIdSerializer;
import org.apache.submarine.server.model.database.entities.ModelVersionEntity;
import org.apache.submarine.server.model.database.entities.RegisteredModelEntity;
import org.apache.submarine.server.model.database.service.ModelVersionService;
import org.apache.submarine.server.model.database.service.RegisteredModelService;


public class ModelVersionRestApiTest {
  private ModelVersionRestApi modelVersionRestApi = new ModelVersionRestApi();;
  private final String registered_model_name = "test_registered_model";
  private final String registered_model_description = "test registered model description";
  private final String model_version_description = "test model version description";
  private final String model_version_source = "s3://submarine/test";
  private final String model_version_uid = "test123";
  private final String model_version_experiment_id = "experiment_123";

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
    registeredModel.setName(registered_model_name);
    registeredModel.setDescription(registered_model_description);
    registeredModelService.insert(registeredModel);
    modelVersion1.setName(registered_model_name);
    modelVersion1.setDescription(model_version_description + "1");
    modelVersion1.setVersion(1);
    modelVersion1.setSource(model_version_source + "1");
    modelVersion1.setUserId(model_version_uid);
    modelVersion1.setExperimentId(model_version_experiment_id);
    modelVersionService.insert(modelVersion1);
    modelVersion2.setName(registered_model_name);
    modelVersion2.setDescription(model_version_description + "2");
    modelVersion2.setVersion(2);
    modelVersion2.setSource(model_version_source + "2");
    modelVersion2.setUserId(model_version_uid);
    modelVersion2.setExperimentId(model_version_experiment_id);
    modelVersionService.insert(modelVersion2);
  }

  @Test
  public void testListModelVersion(){
    Response listModelVersionResponse = modelVersionRestApi.listModelVersions(registered_model_name);
    List<ModelVersionEntity> result = getResultListFromResponse(
        listModelVersionResponse, ModelVersionEntity.class);
    assertEquals(result.size(), 2);
    verifyResult(result.get(0), modelVersion1);
    verifyResult(result.get(1), modelVersion2);
  }

  @Test
  public void testGetModelVersion(){
    Response getModelVersionResponse = modelVersionRestApi.getModelVersion(registered_model_name, 1);
    ModelVersionEntity result = getResultFromResponse(getModelVersionResponse, ModelVersionEntity.class);
    verifyResult(result, modelVersion1);
  }

  @Test
  public void testDeleteModelVersion(){
    modelVersionRestApi.deleteModelVersion(registered_model_name, 1);
    Response listModelVersionResponse = modelVersionRestApi.listModelVersions(registered_model_name);
    List<ModelVersionEntity> result = getResultListFromResponse(
        listModelVersionResponse, ModelVersionEntity.class);
    assertEquals(result.size(), 1);
    verifyResult(result.get(0), modelVersion2);
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

  private void verifyResult(ModelVersionEntity result, ModelVersionEntity actual){
    assertEquals(result.getName(), actual.getName());
    assertEquals(result.getDescription(), actual.getDescription());
    assertEquals(result.getVersion(), actual.getVersion());
    assertEquals(result.getSource(), actual.getSource());
    assertEquals(result.getExperimentId(), actual.getExperimentId());
  }
}
