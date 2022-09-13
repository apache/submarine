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

import java.io.IOException;
import java.net.URISyntaxException;

import static com.github.tomakehurst.wiremock.client.WireMock.aResponse;
import static com.github.tomakehurst.wiremock.client.WireMock.delete;
import static com.github.tomakehurst.wiremock.client.WireMock.get;
import static com.github.tomakehurst.wiremock.client.WireMock.post;
import static com.github.tomakehurst.wiremock.client.WireMock.urlPathEqualTo;
import static org.apache.submarine.server.submitter.k8s.client.K8sMockClient.getResourceFileContent;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

public class SubmitterTensorflowApiTest {

  private static final Logger LOG = LoggerFactory.getLogger(SubmitterTensorflowApiTest.class);

  private K8sSubmitter submitter;

  private static final String namespace = "default";
  private static final String experimentId = "experiment-1659167632755-0001";
  private ExperimentSpec experimentSpec;

  @Rule
  public WireMockRule wireMockRule = K8sMockClient.getWireMockRule();

  @Before
  public void setup() throws IOException, URISyntaxException {
    experimentSpec = SubmitterFileUtil.buildFromJsonFile(experimentId, SpecBuilder.tfJobReqFile);

    // save tf url
    MappingBuilder tfPost = post(urlPathEqualTo(
        "/apis/kubeflow.org/v1/namespaces/default/tfjobs"))
        .withHeader("Content-Type", new EqualToPattern("application/json; charset=UTF-8"))
        .willReturn(
            aResponse()
                .withStatus(200)
                .withBody(getResourceFileContent("client/experiment/tf-read-api.json")));
    // save pod agent url
    String agentName = AgentPod.getNormalizePodName(CustomResourceType.TFJob, experimentId);
    MappingBuilder agentPost = post(urlPathEqualTo("/api/v1/namespaces/default/pods"))
        .withHeader("Content-Type", new EqualToPattern("application/json; charset=UTF-8"))
        .willReturn(
            aResponse()
                .withStatus(200)
                .withBody("{\"metadata\":{\"name\":\"" + agentName + "\"," + "\"namespace\":\"default\"}}"));

    // get tf url
    MappingBuilder tfGet = get(urlPathEqualTo(
        MockClientUtil.getTfJobUrl(namespace, experimentId)))
        .withHeader("Content-Type", new EqualToPattern("application/json"))
        .willReturn(
            aResponse()
                .withStatus(200)
                .withBody(getResourceFileContent("client/experiment/tf-read-api.json")));

    //  delete tf url
    MappingBuilder tfDelete = delete(urlPathEqualTo(
        MockClientUtil.getTfJobUrl(namespace, experimentId)))
        .withHeader("Content-Type", new EqualToPattern("application/json; charset=UTF-8"))
        .willReturn(
            aResponse()
                .withStatus(200)
                .withBody(getResourceFileContent("client/experiment/tf-delete-api.json")));
    //  delete agent pod url
    MappingBuilder agentDelete = delete(urlPathEqualTo(MockClientUtil.getPodUrl(namespace, agentName)))
        .withHeader("Content-Type", new EqualToPattern("application/json; charset=UTF-8"))
        .willReturn(
            aResponse()
                .withStatus(200)
                .withBody(MockClientUtil.getMockSuccessStatus(agentName)));

    K8sClient k8sClient = new K8sMockClient(tfPost, agentPost, tfGet, tfDelete, agentDelete);
    try {
      submitter = new K8sSubmitter(k8sClient);
      submitter.initialize(null);
    } catch (Exception e) {
      LOG.warn("Init K8sSubmitter failed, but we can continue", e);
    }
  }

  @Test
  public void testCreateTensorflowJob() {
    // create tensorflow
    Experiment experiment = submitter.createExperiment(experimentSpec);
    // check return value
    assertNotNull(experiment);
    assertNotNull(experiment.getUid());
    assertEquals(experiment.getStatus(), "Running");
  }

  @Test
  public void testFindTensorflowJob() {
    // get tensorflow
    Experiment experiment = submitter.findExperiment(experimentSpec);
    // check return value
    assertNotNull(experiment);
    assertNotNull(experiment.getUid());
    assertEquals("status is not running", experiment.getStatus(), "Running");
  }

  @Test
  public void testDeleteTensorflowJob() {
    // delete tensorflow
    Experiment experiment = submitter.deleteExperiment(experimentSpec);
    // check return value
    assertNotNull(experiment);
  }
}
