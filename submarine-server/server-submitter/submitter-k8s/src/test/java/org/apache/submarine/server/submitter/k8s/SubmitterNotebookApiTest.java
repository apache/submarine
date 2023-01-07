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

package org.apache.submarine.server.submitter.k8s;

import com.github.tomakehurst.wiremock.client.MappingBuilder;
import com.github.tomakehurst.wiremock.junit.WireMockRule;
import com.github.tomakehurst.wiremock.matching.EqualToPattern;
import org.apache.submarine.commons.utils.SubmarineConfiguration;
import org.apache.submarine.server.api.notebook.Notebook;
import org.apache.submarine.server.api.spec.EnvironmentSpec;
import org.apache.submarine.server.api.spec.NotebookMeta;
import org.apache.submarine.server.api.spec.NotebookPodSpec;
import org.apache.submarine.server.api.spec.NotebookSpec;
import org.apache.submarine.server.submitter.k8s.client.K8sClient;
import org.apache.submarine.server.submitter.k8s.client.K8sMockClient;
import org.apache.submarine.server.submitter.k8s.client.MockClientUtil;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

import static com.github.tomakehurst.wiremock.client.WireMock.aResponse;
import static com.github.tomakehurst.wiremock.client.WireMock.delete;
import static com.github.tomakehurst.wiremock.client.WireMock.get;
import static com.github.tomakehurst.wiremock.client.WireMock.post;
import static com.github.tomakehurst.wiremock.client.WireMock.urlPathEqualTo;
import static org.apache.submarine.server.submitter.k8s.client.K8sMockClient.getResourceFileContent;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertEquals;

public class SubmitterNotebookApiTest {

  private static final Logger LOG = LoggerFactory.getLogger(SubmitterNotebookApiTest.class);

  private K8sSubmitter submitter;

  private NotebookSpec spec;
  private static final String namespace = "default";
  private static final String notebookId = "notebook_1642402491519_0003";
  private static final String notebookName = "notebook-1642402491519-0003-test-notebook";
  private static final String pvcName = "notebook-pvc-notebook-1642402491519-0003-test-notebook";
  private static final String userPvcName = "notebook-pvc-user-notebook-1642402491519-0003-test-notebook";
  private static final String configmapName = "overwrite-configmap-notebook-1642402491519-0003-test-notebook";

  @Rule
  public WireMockRule wireMockRule = K8sMockClient.getWireMockRule();

  @Before
  public void setup() throws IOException {
    // get notebook url
    MappingBuilder notebookGet = get(urlPathEqualTo(
            MockClientUtil.getNotebookUrl(namespace, notebookName)))
        .withHeader("Content-Type", new EqualToPattern("application/json"))
        .willReturn(
            aResponse()
                .withStatus(200)
                .withBody(getResourceFileContent("client/notebook/notebook-read-api.json")));
    // save endpoint
    //  save pvc url
    MappingBuilder pvcPost = post(urlPathEqualTo("/api/v1/namespaces/default/persistentvolumeclaims"))
        .withHeader("Content-Type", new EqualToPattern("application/json; charset=UTF-8"))
        .willReturn(
            aResponse()
                .withStatus(200)
                .withBody("{\"metadata\":{\"name\":\"test-notebook\",\"namespace\":\"default\"}}"));
    //  save configmap url
    MappingBuilder configmapPost = post(urlPathEqualTo(
        "/api/v1/namespaces/default/configmaps"))
        .withHeader("Content-Type", new EqualToPattern("application/json; charset=UTF-8"))
        .willReturn(
            aResponse()
                .withStatus(200)
                .withBody("{\"metadata\":{\"name\":\"test-notebook-configmap\",\"namespace\":\"default\"}}"));
    //  save notebook url
    MappingBuilder notebookPost = post(urlPathEqualTo("/apis/kubeflow.org/v1/namespaces/default/notebooks"))
        .withHeader("Content-Type", new EqualToPattern("application/json; charset=UTF-8"))
        .willReturn(
            aResponse()
                .withStatus(200)
                .withBody(getResourceFileContent("client/notebook/notebook-read-api.json")));
    //  save istio url
    MappingBuilder istioPost = post(urlPathEqualTo(
        "/apis/networking.istio.io/v1beta1/namespaces/default/virtualservices"))
        .withHeader("Content-Type", new EqualToPattern("application/json; charset=UTF-8"))
        .willReturn(
            aResponse()
                .withStatus(200)
                .withBody("{\"metadata\":{\"name\":\"test-notebook-istio\",\"namespace\":\"default\"}}"));
    // delete endpoint
    //  delete notebook url
    MappingBuilder notebookDelete = delete(urlPathEqualTo(
            MockClientUtil.getNotebookUrl(namespace, notebookName)))
        .withHeader("Content-Type", new EqualToPattern("application/json; charset=UTF-8"))
        .willReturn(
            aResponse()
                .withStatus(200)
                .withBody(getResourceFileContent("client/notebook/notebook-delete-api.json")));
    //  delete istio url
    MappingBuilder istioDelete = delete(urlPathEqualTo(MockClientUtil.getIstioUrl(namespace, notebookName)))
        .withHeader("Content-Type", new EqualToPattern("application/json; charset=UTF-8"))
        .willReturn(
            aResponse()
                .withStatus(200)
                .withBody(MockClientUtil.getMockSuccessStatus(notebookName)));
    //  delete pvc url
    MappingBuilder pvcDelete = delete(urlPathEqualTo(MockClientUtil.getPvcUrl(namespace, pvcName)))
        .withHeader("Content-Type", new EqualToPattern("application/json; charset=UTF-8"))
        .willReturn(
            aResponse()
                .withStatus(200)
                .withBody(MockClientUtil.getMockSuccessStatus(pvcName)));
    //  delete user pvc url
    MappingBuilder userPvcDelete = delete(urlPathEqualTo(MockClientUtil.getPvcUrl(namespace, userPvcName)))
        .withHeader("Content-Type", new EqualToPattern("application/json; charset=UTF-8"))
        .willReturn(
            aResponse()
                .withStatus(200)
                .withBody(MockClientUtil.getMockSuccessStatus(userPvcName)));
    //  delete configmap url
    MappingBuilder configmapDelete = delete(urlPathEqualTo(
            MockClientUtil.getConfigmapUrl(namespace, configmapName)))
        .withHeader("Content-Type", new EqualToPattern("application/json; charset=UTF-8"))
        .willReturn(
            aResponse()
                .withStatus(200)
                .withBody(MockClientUtil.getMockSuccessStatus(configmapName)));
    K8sClient k8sClient = new K8sMockClient(
        notebookGet, // get endpoint
        pvcPost, configmapPost, notebookPost, istioPost, // save endpoint
        notebookDelete, istioDelete, pvcDelete, userPvcDelete, configmapDelete // delete endpoint
    );

    // init notebook spec
    spec = new NotebookSpec();
    NotebookMeta meta = new NotebookMeta();
    meta.setNamespace("default");
    meta.setName("test-notebook");
    spec.setMeta(meta);
    NotebookPodSpec podSpec = new NotebookPodSpec();
    podSpec.setResources("cpu=1,memory=1Gi");
    spec.setSpec(podSpec);
    EnvironmentSpec envSpec = new EnvironmentSpec();
    spec.setEnvironment(envSpec);

    // for test configmap
    SubmarineConfiguration conf = SubmarineConfiguration.getInstance();
    conf.updateConfiguration("submarine.notebook.default.overwrite_json",
            "{\n" +
                    "  \"@jupyterlab/translation-extension:plugin\": {\n" +
                    "    \"locale\": \"zh_CN\"\n" +
                    "  }\n" +
                    "}");

    try {
      submitter = new K8sSubmitter(k8sClient);
      submitter.initialize(null);
    } catch (Exception e) {
      LOG.warn("Init K8sSubmitter failed, but we can continue", e);
    }
  }

  @Test
  public void testFindNotebook() {
    // get notebook
    Notebook notebook = submitter.findNotebook(spec, notebookId);
    // check return value
    assertNotNull(notebook);
    assertNotNull(notebook.getUid());
    assertEquals(notebook.getSpec().getMeta().getNamespace(), "default");
    assertEquals(notebook.getSpec().getMeta().getName(), "test-notebook");
    assertEquals("status is not running", notebook.getStatus(), "running");
  }

  @Test
  public void testCreateNotebook() {
    // create notebook
    Notebook notebook = submitter.createNotebook(spec, notebookId);
    // check return value
    assertNotNull(notebook);
    assertNotNull(notebook.getUid());
    assertEquals(notebook.getName(), "test-notebook");
    assertEquals(notebook.getStatus(), "running");
  }

  @Test
  public void testDeleteNotebook() {
    // delete notebook
    Notebook notebook = submitter.deleteNotebook(spec, notebookId);
    // check return value
    assertNotNull(notebook);
    assertEquals(notebook.getName(), "test-notebook");
  }
}
