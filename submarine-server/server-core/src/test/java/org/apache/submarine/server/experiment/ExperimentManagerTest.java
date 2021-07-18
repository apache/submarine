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

package org.apache.submarine.server.experiment;


import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import org.apache.submarine.commons.runtime.exception.SubmarineException;
import org.apache.submarine.commons.utils.exception.SubmarineRuntimeException;
import org.apache.submarine.server.api.Submitter;
import org.apache.submarine.server.api.experiment.Experiment;
import org.apache.submarine.server.api.experiment.ExperimentId;
import org.apache.submarine.server.api.spec.ExperimentSpec;
import org.apache.submarine.server.experiment.database.ExperimentEntity;
import org.apache.submarine.server.experiment.database.ExperimentService;
import org.junit.Before;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.Reader;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.doReturn;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.when;


public class ExperimentManagerTest {
  private final Logger LOG = LoggerFactory.getLogger(ExperimentManagerTest.class);
  private ExperimentManager experimentManager;
  private Submitter mockSubmitter;
  private ExperimentService mockService;
  private String specFile = "/experiment/spec.json";
  private String newSpecFile = "/experiment/new_spec.json";
  private String resultFile = "/experiment/result.json";
  private String statusFile = "/experiment/status.json";

  private ExperimentSpec spec;
  private ExperimentSpec newSpec;
  private Experiment result;
  private Experiment status;


  @Before
  public void init() throws SubmarineException {
    spec = (ExperimentSpec) buildFromJsonFile(ExperimentSpec.class, specFile);
    newSpec = (ExperimentSpec) buildFromJsonFile(ExperimentSpec.class, newSpecFile);
    result = (Experiment) buildFromJsonFile(Experiment.class, resultFile);
    status = (Experiment) buildFromJsonFile(Experiment.class, statusFile);

    mockSubmitter = mock(Submitter.class);
    mockService = mock(ExperimentService.class);
    experimentManager = new ExperimentManager(mockSubmitter, mockService);
  }

  @Test
  public void testCreateExperiment() {

    // Create a experimentID for this experiment
    ExperimentId experimentId = new ExperimentId();
    experimentId.setServerTimestamp(System.currentTimeMillis());
    experimentId.setId(1);

    // Construct expected result
    Experiment expectedExperiment = new Experiment();
    expectedExperiment.setSpec(spec);
    expectedExperiment.setExperimentId(experimentId);
    expectedExperiment.rebuild(result);

    // Spy experimentManager in order to stub generateExperimentId()
    ExperimentManager spyExperimentManager = spy(experimentManager);
    doReturn(experimentId).when(spyExperimentManager).generateExperimentId();

    // Stub mockSubmitter createExperiment
    when(mockSubmitter.createExperiment(any(ExperimentSpec.class))).thenReturn(result);

    // actual experiment should == expected experiment
    Experiment actualExperiment = spyExperimentManager.createExperiment(spec);

    verifyResult(expectedExperiment, actualExperiment);
  }

  @Test
  public void testFinaExperimentByTag() {

    ExperimentId experimentId1 = new ExperimentId();
    experimentId1.setServerTimestamp(System.currentTimeMillis());
    experimentId1.setId(1);

    Experiment experiment1 = new Experiment();
    experiment1.setSpec(spec);
    experiment1.setExperimentId(experimentId1);
    experiment1.rebuild(result);

    ExperimentId experimentId2 = new ExperimentId();
    experimentId2.setServerTimestamp(System.currentTimeMillis());
    experimentId2.setId(2);

    ExperimentEntity entity1 = new ExperimentEntity();
    entity1.setId(experiment1.getExperimentId().toString());
    entity1.setExperimentSpec(new GsonBuilder().disableHtmlEscaping().create().toJson(experiment1.getSpec()));

    doReturn(Arrays.asList(entity1)).when(mockService).selectAll();

    when(mockSubmitter.findExperiment(any(ExperimentSpec.class))).thenReturn(experiment1);

    ExperimentManager spyExperimentManager = spy(experimentManager);
    List<Experiment> foundExperiments = spyExperimentManager.listExperimentsByTag("stable");
    assertEquals(1, foundExperiments.size());
    List<Experiment> foundExperiments2 = spyExperimentManager.listExperimentsByTag("test");
    assertEquals(0, foundExperiments2.size());
  }

  @Test
  public void testGetExperiment() {

    // Create the experimentID for this experiment
    ExperimentId experimentId = new ExperimentId();
    experimentId.setServerTimestamp(System.currentTimeMillis());
    experimentId.setId(1);

    // Create the entity
    ExperimentEntity entity = new ExperimentEntity();
    entity.setExperimentSpec(toJson(spec));
    entity.setId(experimentId.toString());

    // Construct expected result
    Experiment expectedExperiment = new Experiment();
    expectedExperiment.setSpec(spec);
    expectedExperiment.setExperimentId(experimentId);
    expectedExperiment.rebuild(result);


    // Stub service select
    // Pretend there is a entity in db
    when(mockService.select(any(String.class))).thenReturn(entity);

    // Stub mockSubmitter findExperiment
    when(mockSubmitter.findExperiment(any(ExperimentSpec.class))).thenReturn(result);

    // get experiment
    Experiment actualExperiment = experimentManager.getExperiment(experimentId.toString());

    verifyResult(expectedExperiment, actualExperiment);
  }

  @Test
  public void testPatchExperiment() {

    // Create the experimentID for this experiment
    ExperimentId experimentId = new ExperimentId();
    experimentId.setServerTimestamp(System.currentTimeMillis());
    experimentId.setId(1);

    // Create the entity
    ExperimentEntity entity = new ExperimentEntity();
    entity.setExperimentSpec(toJson(spec));
    entity.setId(experimentId.toString());

    // Construct expected result
    Experiment expectedExperiment = new Experiment();
    expectedExperiment.setSpec(newSpec);
    expectedExperiment.setExperimentId(experimentId);
    expectedExperiment.rebuild(result);


    // Stub service select
    // Pretend there is a entity in db
    when(mockService.select(any(String.class))).thenReturn(entity);

    // Stub mockSubmitter patchExperiment
    when(mockSubmitter.patchExperiment(any(ExperimentSpec.class))).thenReturn(result);

    // patch experiment
    Experiment actualExperiment = experimentManager.patchExperiment(experimentId.toString(), newSpec);

    verifyResult(expectedExperiment, actualExperiment);
  }

  @Test
  public void testDeleteExperiment() {

    // Create the experimentID for this experiment
    ExperimentId experimentId = new ExperimentId();
    experimentId.setServerTimestamp(System.currentTimeMillis());
    experimentId.setId(1);

    // Create the entity
    ExperimentEntity entity = new ExperimentEntity();
    entity.setExperimentSpec(toJson(spec));
    entity.setId(experimentId.toString());

    // Construct expected result
    Experiment expectedExperiment = new Experiment();
    expectedExperiment.setSpec(spec);
    expectedExperiment.setExperimentId(experimentId);
    expectedExperiment.rebuild(status);


    // Stub service select
    // Pretend there is a entity in db
    when(mockService.select(any(String.class))).thenReturn(entity);

    // Stub mockSubmitter deleteExperiment
    when(mockSubmitter.deleteExperiment(any(ExperimentSpec.class))).thenReturn(status);

    // delete experiment
    Experiment actualExperiment = experimentManager.deleteExperiment(experimentId.toString());

    verifyResult(expectedExperiment, actualExperiment);
  }

  @Test(expected = SubmarineRuntimeException.class)
  public void testGetNotFoundExperiment() {
    // Create the experimentID for this experiment
    ExperimentId experimentId = new ExperimentId();
    experimentId.setServerTimestamp(System.currentTimeMillis());
    experimentId.setId(1);

    // Stub service select
    // Pretend that we cannot find the entity
    when(mockService.select(any(String.class))).thenReturn(null);

    // get experiment
    experimentManager.getExperiment(experimentId.toString());
  }

  private void verifyResult(Experiment expected, Experiment actual) {
    assertEquals(expected.getUid(), actual.getUid());
    assertEquals(expected.getCreatedTime(), actual.getCreatedTime());
    assertEquals(expected.getRunningTime(), actual.getRunningTime());
    assertEquals(expected.getAcceptedTime(), actual.getAcceptedTime());
    assertEquals(expected.getName(), actual.getName());
    assertEquals(expected.getStatus(), actual.getStatus());
    assertEquals(expected.getExperimentId(), actual.getExperimentId());
    assertEquals(expected.getFinishedTime(), actual.getFinishedTime());
    assertEquals(expected.getSpec().getMeta().getName(), actual.getSpec().getMeta().getName());
    assertEquals(expected.getSpec().getMeta().getFramework(), actual.getSpec().getMeta().getFramework());
    assertEquals(expected.getSpec().getMeta().getNamespace(), actual.getSpec().getMeta().getNamespace());
    assertEquals(
        expected.getSpec().getEnvironment().getImage(),
        actual.getSpec().getEnvironment().getImage())
    ;

    assertEquals(expected.getSpec().getMeta().getTags().toString(), actual.getSpec().getMeta().getTags().toString());
  }

  private Object buildFromJsonFile(Object obj, String filePath) throws SubmarineException {
    Gson gson = new GsonBuilder().create();
    try (Reader reader = Files.newBufferedReader(getCustomJobSpecFile(filePath).toPath(),
      StandardCharsets.UTF_8)) {
      if (obj.equals(ExperimentSpec.class)) {
        return gson.fromJson(reader, ExperimentSpec.class);
      } else {
        return gson.fromJson(reader, Experiment.class);
      }
    } catch (Exception e) {
      LOG.error(e.getMessage());
      throw new SubmarineException(e.getMessage());
    }
  }

  private File getCustomJobSpecFile(String path) throws URISyntaxException {
    URL fileUrl = this.getClass().getResource(path);
    return new File(fileUrl.toURI());
  }

  private <T> String toJson(T object) {
    return new GsonBuilder().disableHtmlEscaping().create().toJson(object);
  }
}
