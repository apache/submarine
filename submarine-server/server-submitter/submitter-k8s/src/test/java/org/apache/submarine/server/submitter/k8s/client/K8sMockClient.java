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

package org.apache.submarine.server.submitter.k8s.client;

import com.github.tomakehurst.wiremock.client.MappingBuilder;
import com.github.tomakehurst.wiremock.junit.WireMockRule;
import io.kubernetes.client.openapi.ApiClient;
import io.kubernetes.client.openapi.apis.AppsV1Api;
import io.kubernetes.client.openapi.apis.CoreV1Api;
import io.kubernetes.client.openapi.apis.CustomObjectsApi;
import io.kubernetes.client.openapi.models.CoreV1Event;
import io.kubernetes.client.openapi.models.CoreV1EventList;
import io.kubernetes.client.openapi.models.V1ConfigMap;
import io.kubernetes.client.openapi.models.V1ConfigMapList;
import io.kubernetes.client.openapi.models.V1PersistentVolumeClaim;
import io.kubernetes.client.openapi.models.V1PersistentVolumeClaimList;
import io.kubernetes.client.openapi.models.V1Pod;
import io.kubernetes.client.openapi.models.V1PodList;
import io.kubernetes.client.util.ClientBuilder;
import io.kubernetes.client.util.generic.GenericKubernetesApi;
import org.apache.submarine.serve.istio.IstioVirtualService;
import org.apache.submarine.serve.istio.IstioVirtualServiceList;
import org.apache.submarine.serve.seldon.SeldonDeployment;
import org.apache.submarine.serve.seldon.SeldonDeploymentList;
import org.apache.submarine.serve.utils.IstioConstants;
import org.apache.submarine.server.submitter.k8s.model.notebook.NotebookCR;
import org.apache.submarine.server.submitter.k8s.model.notebook.NotebookCRList;
import org.apache.submarine.server.submitter.k8s.model.pytorchjob.PyTorchJob;
import org.apache.submarine.server.submitter.k8s.model.pytorchjob.PyTorchJobList;
import org.apache.submarine.server.submitter.k8s.model.tfjob.TFJob;
import org.apache.submarine.server.submitter.k8s.model.tfjob.TFJobList;
import org.apache.submarine.server.submitter.k8s.model.xgboostjob.XGBoostJob;
import org.apache.submarine.server.submitter.k8s.model.xgboostjob.XGBoostJobList;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Objects;

import static com.github.tomakehurst.wiremock.client.WireMock.aResponse;
import static com.github.tomakehurst.wiremock.client.WireMock.get;
import static com.github.tomakehurst.wiremock.client.WireMock.urlPathEqualTo;

public class K8sMockClient implements K8sClient {

  public static String getResourceFileContent(String resource) {
    File file = new File(Objects.requireNonNull(
            K8sMockClient.class.getClassLoader().getResource(resource)).getPath()
    );
    try {
      return new String(Files.readAllBytes(Paths.get(file.toString())));
    } catch (IOException e) {
      e.printStackTrace();
      return null;
    }
  }

  private static final String DISCOVERY_API = getResourceFileContent("client/discovery-api.json");
  private static final String DISCOVERY_APIV1 = getResourceFileContent("client/discovery-api-v1.json");
  private static final String DISCOVERY_APIS = getResourceFileContent("client/discovery-apis.json");

  private final ApiClient apiClient;
  private final CoreV1Api coreApi;
  private final AppsV1Api appsV1Api;
  private final CustomObjectsApi customObjectsApi;
  private final GenericKubernetesApi<V1ConfigMap, V1ConfigMapList> configMapClient;
  private final GenericKubernetesApi<NotebookCR, NotebookCRList> notebookCRClient;
  private final GenericKubernetesApi<V1PersistentVolumeClaim, V1PersistentVolumeClaimList>
          persistentVolumeClaimClient;
  private final GenericKubernetesApi<IstioVirtualService, IstioVirtualServiceList> istioVirtualServiceClient;
  private final GenericKubernetesApi<V1Pod, V1PodList> podClient;

  // train operator client
  private final GenericKubernetesApi<TFJob, TFJobList> tfJobClient;
  private final GenericKubernetesApi<PyTorchJob, PyTorchJobList> pyTorchJobClient;
  private final GenericKubernetesApi<XGBoostJob, XGBoostJobList> xgboostJobClient;

  private static final WireMockRule wireMockRule = new WireMockRule(8384);

  public static WireMockRule getWireMockRule() {
    return wireMockRule;
  }

  public K8sMockClient() throws IOException {
    apiClient = new ClientBuilder().setBasePath("http://localhost:" + 8384).build();
    wireMockRule.stubFor(
            get(urlPathEqualTo("/api"))
                    .willReturn(
                            aResponse()
                                    .withStatus(200)
                                    .withBody(DISCOVERY_API)));
    wireMockRule.stubFor(
            get(urlPathEqualTo("/apis"))
                    .willReturn(
                            aResponse()
                                    .withStatus(200)
                                    .withBody(DISCOVERY_APIS)));
    wireMockRule.stubFor(
            get(urlPathEqualTo("/api/v1"))
                    .willReturn(
                            aResponse()
                                    .withStatus(200)
                                    .withBody(DISCOVERY_APIV1)));
    coreApi = new CoreV1Api();
    appsV1Api = new AppsV1Api();
    customObjectsApi = new CustomObjectsApi();
    configMapClient =
            new GenericKubernetesApi<>(
                    V1ConfigMap.class, V1ConfigMapList.class,
                    "", "v1", "configmaps", apiClient);
    notebookCRClient =
            new GenericKubernetesApi<>(
                    NotebookCR.class, NotebookCRList.class,
                    NotebookCR.CRD_NOTEBOOK_GROUP_V1, NotebookCR.CRD_NOTEBOOK_VERSION_V1,
                    NotebookCR.CRD_NOTEBOOK_PLURAL_V1, apiClient);
    persistentVolumeClaimClient =
            new GenericKubernetesApi<>(
                    V1PersistentVolumeClaim.class, V1PersistentVolumeClaimList.class,
                    "", "v1", "persistentvolumeclaims", apiClient);
    istioVirtualServiceClient =
            new GenericKubernetesApi<>(
                    IstioVirtualService.class, IstioVirtualServiceList.class,
                    IstioConstants.GROUP, IstioConstants.VERSION,
                    IstioConstants.PLURAL, apiClient);
    podClient =
            new GenericKubernetesApi<>(
                    V1Pod.class, V1PodList.class,
                    "", "v1", "pods", apiClient);

    tfJobClient =
            new GenericKubernetesApi<>(
                    TFJob.class, TFJobList.class,
                    TFJob.CRD_TF_GROUP_V1, TFJob.CRD_TF_VERSION_V1,
                    TFJob.CRD_TF_PLURAL_V1, apiClient);
    pyTorchJobClient =
            new GenericKubernetesApi<>(
                    PyTorchJob.class, PyTorchJobList.class,
                    PyTorchJob.CRD_PYTORCH_GROUP_V1, PyTorchJob.CRD_PYTORCH_VERSION_V1,
                    PyTorchJob.CRD_PYTORCH_PLURAL_V1, apiClient);
    xgboostJobClient =
            new GenericKubernetesApi<>(
                    XGBoostJob.class, XGBoostJobList.class,
                    XGBoostJob.CRD_XGBOOST_GROUP_V1, XGBoostJob.CRD_XGBOOST_VERSION_V1,
                    XGBoostJob.CRD_XGBOOST_PLURAL_V1, apiClient);
  }

  public K8sMockClient(MappingBuilder... mappingBuilders) throws IOException {
    this();
    addMappingBuilders(mappingBuilders);
  }

  public void addMappingBuilders(MappingBuilder... mappingBuilders) {
    // add MappingBuilder to WireMockRule
    for (MappingBuilder mappingBuilder : mappingBuilders) {
      wireMockRule.stubFor(mappingBuilder);
    }
  }

  public ApiClient getApiClient() {
    return apiClient;
  }

  public CustomObjectsApi getCustomObjectsApi() {
    return customObjectsApi;
  }

  public CoreV1Api getCoreApi() {
    return coreApi;
  }

  public AppsV1Api getAppsV1Api() {
    return appsV1Api;
  }

  @Override
  public GenericKubernetesApi<V1Pod, V1PodList> getPodClient() {
    return podClient;
  }

  @Override
  public GenericKubernetesApi<CoreV1Event, CoreV1EventList> getEventClient() {
    return null;
  }

  @Override
  public GenericKubernetesApi<V1PersistentVolumeClaim, V1PersistentVolumeClaimList>
      getPersistentVolumeClaimClient() {
    return persistentVolumeClaimClient;
  }

  @Override
  public GenericKubernetesApi<V1ConfigMap, V1ConfigMapList> getConfigMapClient() {
    return configMapClient;
  }

  @Override
  public GenericKubernetesApi<TFJob, TFJobList> getTfJobClient() {
    return tfJobClient;
  }

  @Override
  public GenericKubernetesApi<PyTorchJob, PyTorchJobList> getPyTorchJobClient() {
    return pyTorchJobClient;
  }

  @Override
  public GenericKubernetesApi<XGBoostJob, XGBoostJobList> getXGBoostJobClient() {
    return xgboostJobClient;
  }

  @Override
  public GenericKubernetesApi<NotebookCR, NotebookCRList> getNotebookCRClient() {
    return notebookCRClient;
  }

  @Override
  public GenericKubernetesApi<SeldonDeployment, SeldonDeploymentList> getSeldonDeploymentClient() {
    return null;
  }

  @Override
  public GenericKubernetesApi<IstioVirtualService, IstioVirtualServiceList> getIstioVirtualServiceClient() {
    return istioVirtualServiceClient;
  }
}
