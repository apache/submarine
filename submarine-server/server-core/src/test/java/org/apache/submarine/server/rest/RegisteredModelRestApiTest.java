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
import org.apache.submarine.server.model.database.entities.RegisteredModelEntity;
import org.apache.submarine.server.model.database.service.RegisteredModelService;

public class RegisteredModelRestApiTest {
  private final RegisteredModelService registeredModelService = new RegisteredModelService();
  private final String registered_model_name = "test_registered_model";
  private final String registered_model_description = "test registered model description";
  private static final GsonBuilder gsonBuilder = new GsonBuilder()
      .registerTypeAdapter(ExperimentId.class, new ExperimentIdSerializer())
      .registerTypeAdapter(ExperimentId.class, new ExperimentIdDeserializer());
  private static Gson gson = gsonBuilder.setDateFormat("yyyy-MM-dd HH:mm:ss").create();
  private RegisteredModelEntity registeredModel = new RegisteredModelEntity();

  private final RegisteredModelRestApi registeredModelRestApi = new RegisteredModelRestApi();


  @Before
  public void testCreateRegisteredModel() {
    registeredModel.setName(registered_model_name);
    registeredModel.setDescription(registered_model_description);
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
    Response getModelVersionResponse = registeredModelRestApi.getRegisteredModel(registered_model_name);
    RegisteredModelEntity result = getResultFromResponse(
        getModelVersionResponse, RegisteredModelEntity.class);
    verifyResult(result, registeredModel);
  }

  @Test
  public void testDeleteRegisteredModel(){
    registeredModelRestApi.deleteRegisteredModel(registered_model_name);
    Response listModelVersionResponse = registeredModelRestApi.listRegisteredModels();
    List<RegisteredModelEntity> result = getResultListFromResponse(
        listModelVersionResponse, RegisteredModelEntity.class);
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
