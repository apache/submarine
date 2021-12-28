package org.apache.submarine.server.rest;

import static org.junit.Assert.assertEquals;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import javax.ws.rs.core.Response;

import org.apache.submarine.server.api.common.CustomResourceType;
import org.apache.submarine.server.api.experiment.ExperimentId;
import org.apache.submarine.server.gson.ExperimentIdDeserializer;
import org.apache.submarine.server.gson.ExperimentIdSerializer;
import org.apache.submarine.server.internal.InternalServiceManager;
import org.apache.submarine.server.response.JsonResponse;
import org.junit.Before;
import org.junit.Test;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

public class InternalServiceRestApiTest {
    
  InternalServiceRestApi internalServiceRestApi;
  private static final GsonBuilder gsonBuilder = new GsonBuilder()
          .registerTypeAdapter(ExperimentId.class, new ExperimentIdSerializer())
          .registerTypeAdapter(ExperimentId.class, new ExperimentIdDeserializer());
  private static Gson gson = gsonBuilder.setDateFormat("yyyy-MM-dd HH:mm:ss").create();
  
  @Before
  public void init() {
    InternalServiceManager internalServiceManager = mock(InternalServiceManager.class);
    internalServiceRestApi = new InternalServiceRestApi();
    internalServiceRestApi.setInternalServiceManager(internalServiceManager);
  }
  
  @Test
  public void testUpdateCRStatus() {
    when(internalServiceRestApi.updateEnvironment(any(String.class),
        any(String.class), any(String.class))).thenReturn(new JsonResponse.
        Builder<String>(Response.Status.OK).
        success(true).build());
    
    Response reponse = internalServiceRestApi.updateEnvironment(CustomResourceType.
        Notebook.getCustomResourceType(), "notebookId", "running");
    
    assertEquals(getResultFromResponse(reponse, String.class), Response.Status.OK.toString());
    
  }
  
  private <T> T getResultFromResponse(Response response, Class<T> typeT) {
    String entity = (String) response.getEntity();
    JsonObject object = new JsonParser().parse(entity).getAsJsonObject();
    JsonElement result = object.get("result");
    return gson.fromJson(result, typeT);
  }
}
