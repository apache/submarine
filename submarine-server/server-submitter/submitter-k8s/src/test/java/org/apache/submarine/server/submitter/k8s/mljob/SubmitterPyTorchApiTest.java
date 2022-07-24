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

package org.apache.submarine.server.submitter.k8s.mljob;

import com.github.tomakehurst.wiremock.client.MappingBuilder;
import com.github.tomakehurst.wiremock.junit.WireMockRule;
import com.github.tomakehurst.wiremock.matching.EqualToPattern;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import org.apache.submarine.server.api.common.CustomResourceType;
import org.apache.submarine.server.api.experiment.Experiment;
import org.apache.submarine.server.api.spec.ExperimentSpec;
import org.apache.submarine.server.submitter.k8s.K8sSubmitter;
import org.apache.submarine.server.submitter.k8s.SpecBuilder;
import org.apache.submarine.server.submitter.k8s.client.K8sClient;
import org.apache.submarine.server.submitter.k8s.client.K8sMockClient;
import org.apache.submarine.server.submitter.k8s.client.MockClientUtil;
import org.apache.submarine.server.submitter.k8s.model.AgentPod;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.io.Reader;
import java.net.URISyntaxException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;

import static com.github.tomakehurst.wiremock.client.WireMock.aResponse;
import static com.github.tomakehurst.wiremock.client.WireMock.delete;
import static com.github.tomakehurst.wiremock.client.WireMock.get;
import static com.github.tomakehurst.wiremock.client.WireMock.post;
import static com.github.tomakehurst.wiremock.client.WireMock.urlPathEqualTo;
import static org.apache.submarine.server.submitter.k8s.client.K8sMockClient.getResourceFileContent;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

public class SubmitterPyTorchApiTest extends SpecBuilder {

  private static final Logger LOG = LoggerFactory.getLogger(SubmitterPyTorchApiTest.class);

  private K8sSubmitter submitter;

  private static final String namespace = "default";
  private final String experimentId = "experiment-1658656463509-0001";
  private ExperimentSpec experimentSpec;

  @Rule
  public WireMockRule wireMockRule = K8sMockClient.getWireMockRule();

  protected Object buildFromJsonFile(Object obj, String filePath) throws IOException,
          URISyntaxException {
    Gson gson = new GsonBuilder().create();
    try (Reader reader = Files.newBufferedReader(
        new File(this.getClass().getResource(filePath).toURI()).toPath(),
        StandardCharsets.UTF_8)) {
      ExperimentSpec experimentSpec = gson.fromJson(reader, ExperimentSpec.class);
      experimentSpec.getMeta().setExperimentId(experimentId);
      return experimentSpec;
    }
  }

  @Before
  public void setup() throws IOException, URISyntaxException {
    experimentSpec = (ExperimentSpec) buildFromJsonFile(ExperimentSpec.class, pytorchJobReqFile);

    // save pytorch url
    MappingBuilder pytorchPost = post(urlPathEqualTo(
            "/apis/kubeflow.org/v1/namespaces/default/pytorchjobs"))
        .withHeader("Content-Type", new EqualToPattern("application/json; charset=UTF-8"))
        .willReturn(
            aResponse()
              .withStatus(200)
              .withBody(getResourceFileContent("client/experiment/pytorch-read-api.json")));
    // save pod agent url
    MappingBuilder agentPost = post(urlPathEqualTo("/api/v1/namespaces/default/pods"))
        .withHeader("Content-Type", new EqualToPattern("application/json; charset=UTF-8"))
        .willReturn(
            aResponse()
                .withStatus(200)
                .withBody("{\"metadata\":" +
                    "{\"name\":\"experiment-1658656683252-0001-tfjob-tensorflow-dist-mnist-agent\"," +
                    "\"namespace\":\"default\"}}"));
    // get pytorch url
    MappingBuilder pytorchGet = get(urlPathEqualTo(
        MockClientUtil.getPytorchJobUrl(namespace, experimentId)))
        .withHeader("Content-Type", new EqualToPattern("application/json"))
        .willReturn(
            aResponse()
                .withStatus(200)
                .withBody(getResourceFileContent("client/experiment/pytorch-read-api.json")));

    //  delete pytorch url
    MappingBuilder pytorchDelete = delete(urlPathEqualTo(
        MockClientUtil.getPytorchJobUrl(namespace, experimentId)))
        .withHeader("Content-Type", new EqualToPattern("application/json; charset=UTF-8"))
        .willReturn(
            aResponse()
                .withStatus(200)
                .withBody(getResourceFileContent("client/experiment/pytorch-delete-api.json")));
    //  delete agent pod url
    String agentName = AgentPod.getNormalizePodName(
        CustomResourceType.PyTorchJob, "pytorch-dist-mnist", experimentId);
    MappingBuilder podDelete = delete(urlPathEqualTo(MockClientUtil.getPodUrl(namespace, agentName)))
        .withHeader("Content-Type", new EqualToPattern("application/json; charset=UTF-8"))
        .willReturn(
            aResponse()
                .withStatus(200)
                .withBody(MockClientUtil.getMockSuccessStatus(agentName)));

    K8sClient k8sClient = new K8sMockClient(pytorchPost, agentPost, pytorchGet, pytorchDelete, podDelete);
    try {
      submitter = new K8sSubmitter(k8sClient);
      submitter.initialize(null);
    } catch (Exception e) {
      LOG.warn("Init K8sSubmitter failed, but we can continue", e);
    }
  }

  @Test
  public void testCreatePyTorchJob() {
    // create pytorch
    Experiment experiment = submitter.createExperiment(experimentSpec);
    // check return value
    assertNotNull(experiment);
    assertNotNull(experiment.getUid());
    assertEquals(experiment.getStatus(), "Running");
  }

  @Test
  public void testFindPyTorchJob() {
    // get pytorch
    Experiment experiment = submitter.findExperiment(experimentSpec);
    // check return value
    assertNotNull(experiment);
    assertNotNull(experiment.getUid());
    assertEquals("status is not running", experiment.getStatus(), "Running");
  }

  @Test
  public void testDeletePyTorchJob() {
    // delete pytorch
    Experiment experiment = submitter.deleteExperiment(experimentSpec);
    // check return value
    assertNotNull(experiment);
  }
}
