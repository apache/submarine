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
import org.apache.submarine.server.database.experiment.ExperimentManager;
import org.apache.submarine.server.utils.gson.ExperimentIdDeserializer;
import org.apache.submarine.server.utils.gson.ExperimentIdSerializer;
import org.junit.Ignore;
import org.junit.Test;
import org.junit.BeforeClass;
import org.junit.Before;

import javax.ws.rs.core.Response;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.Assert.assertEquals;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class ExperimentRestApiTest {

  private static ExperimentRestApi experimentRestApi;
  private static ExperimentManager mockExperimentManager;
  private final AtomicInteger experimentCounter = new AtomicInteger(0);
  EnvironmentSpec environmentSpec = new EnvironmentSpec();
  KernelSpec kernelSpec = new KernelSpec();
  ExperimentMeta meta = new ExperimentMeta();
  ExperimentSpec experimentSpec = new ExperimentSpec();
  private Experiment actualExperiment;

  private static final GsonBuilder gsonBuilder = new GsonBuilder()
      .registerTypeAdapter(ExperimentId.class, new ExperimentIdSerializer())
      .registerTypeAdapter(ExperimentId.class, new ExperimentIdDeserializer());
  private static Gson gson = gsonBuilder.setDateFormat("yyyy-MM-dd HH:mm:ss").create();

  private static final String experimentAcceptedTime = "2020-08-06T08:39:22.000+08:00";
  private static final String experimentCreatedTime = "2020-08-06T08:39:22.000+08:00";
  private static final String experimentRunningTime = "2020-08-06T08:39:23.000+08:00";
  private static final String experimentFinishedTime = "2020-08-06T08:41:07.000+08:00";
  private static final String experimentName = "tf-example";
  private static final String experimentUid = "0b617cea-81fa-40b6-bbff-da3e400d2be4";
  private static final String experimentStatus = "Succeeded";
  private static final String metaName = "foo";
  private static final String metaFramework = "TensorFlow";
  private static final String dockerImage = "continuumio/anaconda3";
  private static final String kernelSpecName = "team_default_python_3";
  private static final List<String> kernelChannels = Arrays.asList("defaults", "anaconda");
  private static final List<String> kernelCondaDependencies = Arrays.asList(
      "_ipyw_jlab_nb_ext_conf=0.1.0=py37_0",
      "alabaster=0.7.12=py37_0",
      "anaconda=2020.02=py37_0",
      "anaconda-client=1.7.2=py37_0",
      "anaconda-navigator=1.9.12=py37_0");
  private static final List<String> kernelPipDependencies = Arrays.asList(
      "apache-submarine==0.5.0",
      "pyarrow==0.17.0"
  );
  private final ExperimentId experimentId = ExperimentId.newInstance(SubmarineServer.getServerTimeStamp(),
      experimentCounter.incrementAndGet());

  private final String dummyId = "experiment_1597012631706_0001";

  @BeforeClass
  public static void init() {
    mockExperimentManager = mock(ExperimentManager.class);
    experimentRestApi = new ExperimentRestApi();
    experimentRestApi.setExperimentManager(mockExperimentManager);
  }

  @Before
  public void testCreateExperiment() {
    actualExperiment = new Experiment();
    actualExperiment.setAcceptedTime(experimentAcceptedTime);
    actualExperiment.setCreatedTime(experimentCreatedTime);
    actualExperiment.setRunningTime(experimentRunningTime);
    actualExperiment.setFinishedTime(experimentFinishedTime);
    actualExperiment.setUid(experimentUid);
    actualExperiment.setStatus(experimentStatus);
    actualExperiment.setExperimentId(experimentId);
    kernelSpec.setName(kernelSpecName);
    kernelSpec.setChannels(kernelChannels);
    kernelSpec.setCondaDependencies(kernelCondaDependencies);
    kernelSpec.setPipDependencies(kernelPipDependencies);
    meta.setName(metaName);
    meta.setFramework(metaFramework);
    environmentSpec.setDockerImage(dockerImage);
    environmentSpec.setKernelSpec(kernelSpec);
    experimentSpec.setMeta(meta);
    experimentSpec.setEnvironment(environmentSpec);
    actualExperiment.setSpec(experimentSpec);
    when(mockExperimentManager.createExperiment(any(ExperimentSpec.class))).thenReturn(actualExperiment);
    Response createExperimentResponse = experimentRestApi.createExperiment(experimentSpec);
    assertEquals(Response.Status.OK.getStatusCode(), createExperimentResponse.getStatus());
    Experiment result = getResultFromResponse(createExperimentResponse, Experiment.class);
    verifyResult(result, experimentUid);
  }

  @Test
  public void testGetExperiment() {
    when(mockExperimentManager.getExperiment(any(String.class))).thenReturn(actualExperiment);
    Response getExperimentResponse = experimentRestApi.getExperiment(dummyId);
    Experiment result = getResultFromResponse(getExperimentResponse, Experiment.class);
    verifyResult(result, experimentUid);
  }

  @Test
  public void testPatchExperiment() {
    when(mockExperimentManager.patchExperiment(any(String.class), any(ExperimentSpec.class))).
        thenReturn(actualExperiment);
    Response patchExperimentResponse = experimentRestApi.patchExperiment(dummyId, new ExperimentSpec());
    Experiment result = getResultFromResponse(patchExperimentResponse, Experiment.class);
    verifyResult(result, experimentUid);
  }

  @Test
  public void testListLog() {
    List<ExperimentLog> experimentLogList = new ArrayList<>();
    ExperimentLog log1 = new ExperimentLog();
    log1.setExperimentId(dummyId);
    experimentLogList.add(log1);
    when(mockExperimentManager.listExperimentLogsByStatus(any(String.class))).thenReturn(experimentLogList);
    Response listLogResponse = experimentRestApi.listLog("running");
    List<ExperimentLog> result = getResultListFromResponse(listLogResponse, ExperimentLog.class);
    assertEquals(dummyId, result.get(0).getExperimentId());
  }

  @Test
  public void testGetLog() {
    ExperimentLog log1 = new ExperimentLog();
    log1.setExperimentId(dummyId);
    when(mockExperimentManager.getExperimentLog(any(String.class))).thenReturn(log1);
    Response logResponse = experimentRestApi.getLog(dummyId);
    ExperimentLog result = getResultFromResponse(logResponse, ExperimentLog.class);
    assertEquals(dummyId, result.getExperimentId());
  }

  @Test
  public void testListExperiment() {
    Experiment experiment2 = new Experiment();
    experiment2.rebuild(actualExperiment);
    String experiment2Uid = "0b617cea-81fa-40b6-bbff-da3e400d2be5";
    experiment2.setUid(experiment2Uid);
    experiment2.setExperimentId(experimentId);
    List<Experiment> experimentList = new ArrayList<>();
    experimentList.add(actualExperiment);
    experimentList.add(experiment2);
    when(mockExperimentManager.listExperimentsByStatus(any(String.class))).thenReturn(experimentList);
    Response listExperimentResponse = experimentRestApi.listExperiments(Response.Status.OK.toString());
    List<Experiment> result = getResultListFromResponse(listExperimentResponse, Experiment.class);
    verifyResult(result.get(0), experimentUid);
    verifyResult(result.get(1), experiment2Uid);
  }

  // TODO(KUAN-HSUN LI): mock the s3Client
  @Ignore
  @Test
  public void testDeleteExperiment() {
    String log1ID = "experiment_1597012631706_0002";
    when(mockExperimentManager.deleteExperiment(log1ID)).thenReturn(actualExperiment);
    Response deleteExperimentResponse = experimentRestApi.deleteExperiment(log1ID);
    Experiment result = getResultFromResponse(deleteExperimentResponse, Experiment.class);
    verifyResult(result, experimentUid);
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

  private void verifyResult(Experiment experiment, String uid) {
    assertEquals(uid, experiment.getUid());
    assertEquals(experimentCreatedTime, experiment.getCreatedTime());
    assertEquals(experimentRunningTime, experiment.getRunningTime());
    assertEquals(experimentAcceptedTime, experiment.getAcceptedTime());
    assertEquals(experimentStatus, experiment.getStatus());
    assertEquals(experimentId, experiment.getExperimentId());
    assertEquals(experimentFinishedTime, experiment.getFinishedTime());
    assertEquals(metaName, experiment.getSpec().getMeta().getName());
    assertEquals(metaFramework, experiment.getSpec().getMeta().getFramework());
    assertEquals("default", experiment.getSpec().getMeta().getNamespace());
    assertEquals(dockerImage, experiment.getSpec().getEnvironment().getDockerImage());
    assertEquals(kernelChannels, experiment.getSpec().getEnvironment().getKernelSpec().getChannels());
    assertEquals(kernelSpecName, experiment.getSpec().getEnvironment().getKernelSpec().getName());
    assertEquals(kernelCondaDependencies,
        experiment.getSpec().getEnvironment().getKernelSpec().getCondaDependencies());
    assertEquals(kernelPipDependencies,
        experiment.getSpec().getEnvironment().getKernelSpec().getPipDependencies());
  }
}
