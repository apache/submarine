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

import com.google.gson.Gson;
import com.google.gson.JsonElement;
import com.google.gson.JsonArray;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import org.apache.submarine.server.SubmarineServer;
import org.apache.submarine.server.api.experiment.Experiment;
import org.apache.submarine.server.api.experiment.ExperimentId;
import org.apache.submarine.server.api.experiment.ExperimentLog;
import org.apache.submarine.server.api.spec.EnvironmentSpec;
import org.apache.submarine.server.api.spec.ExperimentMeta;
import org.apache.submarine.server.api.spec.ExperimentSpec;
import org.apache.submarine.server.api.spec.KernelSpec;
import org.apache.submarine.server.experiment.ExperimentManager;
import org.apache.submarine.server.gson.ExperimentIdDeserializer;
import org.apache.submarine.server.gson.ExperimentIdSerializer;
import org.junit.Test;
import org.junit.BeforeClass;
import org.junit.Before;
import org.junit.After;

import javax.ws.rs.core.Response;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.Assert.assertEquals;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class ExperimentRestApiTest {

  private static ExperimentRestApi experimentRestApi;
  private static ExperimentManager mockExperimentManager;
  private final AtomicInteger experimentCounter = new AtomicInteger(0);
  private Experiment experiment;

  private static GsonBuilder gsonBuilder = new GsonBuilder()
      .registerTypeAdapter(ExperimentId.class, new ExperimentIdSerializer())
      .registerTypeAdapter(ExperimentId.class, new ExperimentIdDeserializer());
  private static Gson gson = gsonBuilder.setDateFormat("yyyy-MM-dd HH:mm:ss").create();

  private static String experimentAcceptedTime = "2020-08-06T08:39:22.000+08:00";
  private static String experimentCreatedTime = "2020-08-06T08:39:22.000+08:00";
  private static String experimentRunningTime = "2020-08-06T08:39:23.000+08:00";
  private static String experimentFinishedTime = "2020-08-06T08:41:07.000+08:00";
  private static String experimentName = "tf-example";
  private static String experimentUid = "0b617cea-81fa-40b6-bbff-da3e400d2be4";
  private static String experimentStatus = "Succeeded";
  private ExperimentId experimentExperimentId = ExperimentId.newInstance(SubmarineServer.getServerTimeStamp(),
      experimentCounter.incrementAndGet());

  @BeforeClass
  public static void init() {
    mockExperimentManager = mock(ExperimentManager.class);
    experimentRestApi = new ExperimentRestApi();
    experimentRestApi.setExperimentManager(mockExperimentManager);
  }

  @Before
  public void testCreateExperiment() {
    experiment = new Experiment();
    experiment.setAcceptedTime(experimentAcceptedTime);
    experiment.setCreatedTime(experimentCreatedTime);
    experiment.setRunningTime(experimentRunningTime);
    experiment.setFinishedTime(experimentFinishedTime);
    experiment.setUid(experimentUid);
    experiment.setName(experimentName);
    experiment.setStatus(experimentStatus);
    experiment.setExperimentId(experimentExperimentId);
    ExperimentSpec experimentSpec = new ExperimentSpec();
    EnvironmentSpec environmentSpec = new EnvironmentSpec();
    KernelSpec kernelSpec = new KernelSpec();
    ExperimentMeta meta = new ExperimentMeta();
    kernelSpec.setName("team_default_python_3");
    kernelSpec.setChannels(Arrays.asList("defaults", "anaconda"));
    kernelSpec.setDependencies(Arrays.asList(
        "_ipyw_jlab_nb_ext_conf=0.1.0=py37_0",
        "alabaster=0.7.12=py37_0",
        "anaconda=2020.02=py37_0",
        "anaconda-client=1.7.2=py37_0",
        "anaconda-navigator=1.9.12=py37_0"));
    meta.setName("foo");
    meta.setFramework("fooFramework");
    meta.setNamespace("fooNamespace");
    experimentSpec.setEnvironment(environmentSpec);
    environmentSpec.setDockerImage("continuumio/anaconda3");
    environmentSpec.setKernelSpec(kernelSpec);
    experimentSpec.setMeta(meta);
    experiment.setSpec(experimentSpec);
    when(mockExperimentManager.createExperiment(any(ExperimentSpec.class))).thenReturn(experiment);
    Response createExperimentResponse = experimentRestApi.createExperiment(experimentSpec);
    assertEquals(Response.Status.OK.getStatusCode(), createExperimentResponse.getStatus());
    Experiment result = getResultFromResponse(createExperimentResponse, Experiment.class);

    assertEquals(experimentUid, result.getUid());
    assertEquals(experimentCreatedTime, result.getCreatedTime());
    assertEquals(experimentRunningTime, result.getRunningTime());
    assertEquals(experimentAcceptedTime, result.getAcceptedTime());
    assertEquals(experimentName, result.getName());
    assertEquals(experimentStatus, result.getStatus());
    assertEquals(experimentExperimentId, result.getExperimentId());
    assertEquals(experimentFinishedTime, result.getFinishedTime());
  }

  @Test
  public void testGetExperiment() {
    when(mockExperimentManager.getExperiment(any(String.class))).thenReturn(experiment);
    Response getExperimentResponse = experimentRestApi.getExperiment("1");
    Experiment experiment = getResultFromResponse(getExperimentResponse, Experiment.class);
    assertEquals(experimentUid, experiment.getUid());
    assertEquals(experimentCreatedTime, experiment.getCreatedTime());
    assertEquals(experimentRunningTime, experiment.getRunningTime());
    assertEquals(experimentAcceptedTime, experiment.getAcceptedTime());
    assertEquals(experimentName, experiment.getName());
    assertEquals(experimentStatus, experiment.getStatus());
    assertEquals(experimentExperimentId, experiment.getExperimentId());
    assertEquals(experimentFinishedTime, experiment.getFinishedTime());
  }

  @Test
  public void testPatchExperiment() {
    when(mockExperimentManager.patchExperiment(any(String.class), any(ExperimentSpec.class))).
        thenReturn(experiment);
    Response patchExperimentResponse = experimentRestApi.patchExperiment("1", new ExperimentSpec());
    Experiment experiment = getResultFromResponse(patchExperimentResponse, Experiment.class);
    assertEquals(experimentUid, experiment.getUid());
    assertEquals(experimentCreatedTime, experiment.getCreatedTime());
    assertEquals(experimentRunningTime, experiment.getRunningTime());
    assertEquals(experimentAcceptedTime, experiment.getAcceptedTime());
    assertEquals(experimentName, experiment.getName());
    assertEquals(experimentStatus, experiment.getStatus());
    assertEquals(experimentExperimentId, experiment.getExperimentId());
    assertEquals(experimentFinishedTime, experiment.getFinishedTime());
  }

  @Test
  public void testListLog() {
    List<ExperimentLog> experimentLogList = new ArrayList<>();
    ExperimentLog log1 = new ExperimentLog();
    String log1ID = "experiment_1597012631706_0001";
    log1.setExperimentId(log1ID);
    experimentLogList.add(log1);
    when(mockExperimentManager.listExperimentLogsByStatus(any(String.class))).thenReturn(experimentLogList);
    Response listLogResponse = experimentRestApi.listLog("1");
    List<ExperimentLog> logs = getResultListFromResponse(listLogResponse, ExperimentLog.class);
    assertEquals(log1ID, logs.get(0).getExperimentId());
  }

  @Test
  public void testGetLog() {
    ExperimentLog log1 = new ExperimentLog();
    String log1ID = "experiment_1597012631706_0002";
    log1.setExperimentId(log1ID);
    when(mockExperimentManager.getExperimentLog(any(String.class))).thenReturn(log1);
    Response logResponse = experimentRestApi.getLog("1");
    ExperimentLog log = getResultFromResponse(logResponse, ExperimentLog.class);
    assertEquals(log1ID, log.getExperimentId());
  }

  @Test
  public void testListExperiment() {
    Experiment experiment2 = new Experiment();
    String experiment2Uid = "0b617cea-81fa-40b6-bbff-da3e400d2be5";
    experiment2.setUid(experiment2Uid);
    List<Experiment> experimentList = new ArrayList<>();
    experimentList.add(experiment);
    experimentList.add(experiment2);
    when(mockExperimentManager.listExperimentsByStatus(any(String.class))).thenReturn(experimentList);
    Response listExperimentResponse = experimentRestApi.listExperiments(Response.Status.OK.toString());
    List<Experiment> experiments = getResultListFromResponse(listExperimentResponse, Experiment.class);
    assertEquals(experimentUid, experiments.get(0).getUid());
    assertEquals(experiment2Uid, experiments.get(1).getUid());
  }

  @After
  public void testDeleteExperiment() {
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
