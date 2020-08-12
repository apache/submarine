package org.apache.submarine.server.rest;

import com.google.gson.*;
import com.google.gson.internal.LinkedTreeMap;
import com.google.gson.reflect.TypeToken;
import org.apache.submarine.server.SubmarineServer;
import org.apache.submarine.server.api.environment.Environment;
import org.apache.submarine.server.api.experiment.Experiment;
import org.apache.submarine.server.api.experiment.ExperimentId;
import org.apache.submarine.server.api.experiment.ExperimentLog;
import org.apache.submarine.server.api.spec.EnvironmentSpec;
import org.apache.submarine.server.api.spec.ExperimentSpec;
import org.apache.submarine.server.experiment.ExperimentManager;
import org.apache.submarine.server.gson.ExperimentIdDeserializer;
import org.apache.submarine.server.gson.ExperimentIdSerializer;
import org.junit.*;


import javax.ws.rs.core.Response;
import java.util.ArrayList;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

public class ExperimentRestApiTest {

  private static ExperimentRestApi experimentRestApi;
  private static ExperimentManager mockExperimentManager;
  private final AtomicInteger experimentCounter = new AtomicInteger(0);
  private Experiment experiment;

  private static GsonBuilder gsonBuilder = new GsonBuilder()
      .registerTypeAdapter(ExperimentId.class, new ExperimentIdSerializer())
      .registerTypeAdapter(ExperimentId.class, new ExperimentIdDeserializer());
  private static Gson gson = gsonBuilder.setDateFormat("yyyy-MM-dd HH:mm:ss").create();

  @BeforeClass
  public static void init() {
    mockExperimentManager = mock(ExperimentManager.class);
    experimentRestApi = new ExperimentRestApi();
    experimentRestApi.setExperimentManager(mockExperimentManager);
  }

  @Before
  public void createAndUpdateExperiment() {
    experiment = new Experiment();
    experiment.setAcceptedTime("2020-08-06T08:39:22.000+08:00");
    experiment.setCreatedTime("2020-08-06T08:39:22.000+08:00");
    experiment.setRunningTime("2020-08-06T08:39:23.000+08:00");
    experiment.setFinishedTime("2020-08-06T08:41:07.000+08:00");
    experiment.setUid("0b617cea-81fa-40b6-bbff-da3e400d2be4");
    experiment.setName("tf-example");
    experiment.setStatus("Succeeded");
    experiment.setExperimentId(ExperimentId.newInstance(SubmarineServer.getServerTimeStamp(),
        experimentCounter.incrementAndGet()));
    ExperimentSpec experimentSpec = new ExperimentSpec();
    EnvironmentSpec environmentSpec = new EnvironmentSpec();
    environmentSpec.setName("foo");
    experimentSpec.setEnvironment(environmentSpec);
    experiment.setSpec(experimentSpec);
    when(mockExperimentManager.createExperiment(any(ExperimentSpec.class))).thenReturn(experiment);
    Response createExperimentResponse = experimentRestApi.createExperiment(experimentSpec);
    assertEquals(Response.Status.OK.getStatusCode(), createExperimentResponse.getStatus());
    Experiment result = getResultFromResponse(createExperimentResponse, Experiment.class);
    assertEquals(experiment.getAcceptedTime(), result.getAcceptedTime());
  }

  @Test
  public void getExperiment() {
    when(mockExperimentManager.getExperiment(any(String.class))).thenReturn(experiment);
    Response getExperimentResponse = experimentRestApi.getExperiment("1");
    Experiment experiment = getResultFromResponse(getExperimentResponse, Experiment.class);
    assertEquals("0b617cea-81fa-40b6-bbff-da3e400d2be4", experiment.getUid());
    assertEquals("2020-08-06T08:39:22.000+08:00", experiment.getAcceptedTime());
    assertEquals("tf-example", experiment.getName());
    assertEquals("2020-08-06T08:39:22.000+08:00", experiment.getCreatedTime());
    assertEquals("2020-08-06T08:39:23.000+08:00", experiment.getRunningTime());
  }

  @Test
  public void patchExperiment() {
    when(mockExperimentManager.patchExperiment(any(String.class), any(ExperimentSpec.class))).thenReturn(experiment);
    Response patchExperimentResponse = experimentRestApi.patchExperiment("1", new ExperimentSpec());
    Experiment experiment = getResultFromResponse(patchExperimentResponse, Experiment.class);
    assertEquals("0b617cea-81fa-40b6-bbff-da3e400d2be4", experiment.getUid());
    assertEquals("2020-08-06T08:39:22.000+08:00", experiment.getAcceptedTime());
    assertEquals("tf-example", experiment.getName());
    assertEquals("2020-08-06T08:39:22.000+08:00", experiment.getCreatedTime());
    assertEquals("2020-08-06T08:39:23.000+08:00", experiment.getRunningTime());
  }

  @Test
  public void listLog() {
    List<ExperimentLog> experimentLogList = new ArrayList<>();
    ExperimentLog log1 = new ExperimentLog();
    log1.setExperimentId("test id");
    experimentLogList.add(log1);
    when(mockExperimentManager.listExperimentLogsByStatus(any(String.class))).thenReturn(experimentLogList);
    Response listLogResponse = experimentRestApi.listLog("1");
    List<ExperimentLog> logs = getResultListFromResponse(listLogResponse, ExperimentLog.class);
    assertEquals("test id", logs.get(0).getExperimentId());
  }

  @Test
  public void getLog() {
    ExperimentLog log1 = new ExperimentLog();
    log1.setExperimentId("test id");
    when(mockExperimentManager.getExperimentLog(any(String.class))).thenReturn(log1);
    Response logResponse = experimentRestApi.getLog("1");
    ExperimentLog log = getResultFromResponse(logResponse, ExperimentLog.class);
    assertEquals("test id", log.getExperimentId());
  }

  @Test
  public void listExperiment() {
    Experiment experiment2 = new Experiment();
    experiment2.setUid("0b617cea-81fa-40b6-bbff-da3e400d2be5");
    List<Experiment> experimentList = new ArrayList<>();
    experimentList.add(experiment);
    experimentList.add(experiment2);
    when(mockExperimentManager.listExperimentsByStatus(any(String.class))).thenReturn(experimentList);
    Response listExperimentResponse = experimentRestApi.listExperiments(Response.Status.OK.toString());
    List<Experiment> experiments = getResultListFromResponse(listExperimentResponse, Experiment.class);
    assertEquals("0b617cea-81fa-40b6-bbff-da3e400d2be4", experiments.get(0).getUid());
    assertEquals("0b617cea-81fa-40b6-bbff-da3e400d2be5", experiments.get(1).getUid());
  }

  @After
  public void deleteExperiment() {
    when(mockExperimentManager.deleteExperiment("1")).thenReturn(experiment);
    Response deleteExperimentResponse = experimentRestApi.deleteExperiment("1");
    Experiment experiment = getResultFromResponse(deleteExperimentResponse, Experiment.class);
    assertEquals(this.experiment.getAcceptedTime(), experiment.getAcceptedTime());
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
    JsonArray arry = result.getAsJsonArray();
    for (JsonElement jsonElement : arry) {
      list.add(gson.fromJson(jsonElement, typeT));
    }
    return list;
  }
}
