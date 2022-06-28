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

import io.kubernetes.client.openapi.ApiClient;
import io.kubernetes.client.openapi.Configuration;
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
import io.kubernetes.client.util.KubeConfig;
import io.kubernetes.client.util.generic.GenericKubernetesApi;
import org.apache.commons.lang3.StringUtils;
import org.apache.submarine.commons.utils.exception.SubmarineRuntimeException;
import org.apache.submarine.serve.istio.IstioVirtualService;
import org.apache.submarine.serve.istio.IstioVirtualServiceList;
import org.apache.submarine.serve.seldon.SeldonDeployment;
import org.apache.submarine.serve.seldon.SeldonDeploymentList;
import org.apache.submarine.serve.utils.IstioConstants;
import org.apache.submarine.serve.utils.SeldonConstants;
import org.apache.submarine.server.submitter.k8s.model.notebook.NotebookCR;
import org.apache.submarine.server.submitter.k8s.model.notebook.NotebookCRList;
import org.apache.submarine.server.submitter.k8s.model.pytorchjob.PyTorchJob;
import org.apache.submarine.server.submitter.k8s.model.pytorchjob.PyTorchJobList;
import org.apache.submarine.server.submitter.k8s.model.tfjob.TFJob;
import org.apache.submarine.server.submitter.k8s.model.tfjob.TFJobList;
import org.apache.submarine.server.submitter.k8s.model.xgboostjob.XGBoostJob;
import org.apache.submarine.server.submitter.k8s.model.xgboostjob.XGBoostJobList;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.FileReader;
import java.io.IOException;

public class K8sDefaultClient implements K8sClient {

  public static final String KUBECONFIG_ENV = "KUBECONFIG";

  private static final Logger LOG = LoggerFactory.getLogger(K8sDefaultClient.class);

  // K8s API client for CRD
  private final GenericKubernetesApi<V1Pod, V1PodList> podClient;

  private final GenericKubernetesApi<CoreV1Event, CoreV1EventList> eventClient;

  private final GenericKubernetesApi<V1PersistentVolumeClaim, V1PersistentVolumeClaimList>
          persistentVolumeClaimClient;

  private final GenericKubernetesApi<V1ConfigMap, V1ConfigMapList> configMapClient;

  private final GenericKubernetesApi<TFJob, TFJobList> tfJobClient;

  private final GenericKubernetesApi<PyTorchJob, PyTorchJobList> pyTorchJobClient;

  private final GenericKubernetesApi<XGBoostJob, XGBoostJobList> xgboostJobClient;

  private final GenericKubernetesApi<NotebookCR, NotebookCRList> notebookCRClient;

  private final GenericKubernetesApi<SeldonDeployment, SeldonDeploymentList> seldonDeploymentClient;

  private final GenericKubernetesApi<IstioVirtualService, IstioVirtualServiceList> istioVirtualServiceClient;

  private final CoreV1Api coreApi;

  private final AppsV1Api appsV1Api;

  private final CustomObjectsApi customObjectsApi;

  private ApiClient client = null;

  public K8sDefaultClient() {
    String path = System.getenv(KUBECONFIG_ENV);
    if (StringUtils.isNotBlank(path)) {
      try (FileReader reader = new FileReader(path)) {
        LOG.info("init by kubeconfig env path {}", path);
        KubeConfig config = KubeConfig.loadKubeConfig(reader);
        client = ClientBuilder.kubeconfig(config).build();
        Configuration.setDefaultApiClient(client);
      } catch (IOException e) {
        LOG.error(String.format("Initialize K8s submitter failed by kubeconfig env path: %s. %s",
                path, e.getMessage()), e);
      }
    }
    if (client == null) {
      try {
        LOG.info("Maybe in cluster mode, try to initialize the client again.");
        // loading the in-cluster config, including:
        //   1. service-account CA
        //   2. service-account bearer-token
        //   3. service-account namespace
        //   4. master endpoints(ip, port) from pre-set environment variables
        client = ClientBuilder.cluster().build();
        Configuration.setDefaultApiClient(client);
      } catch (IOException e) {
        LOG.error("Initialize K8s submitter failed. " + e.getMessage(), e);
        throw new SubmarineRuntimeException(500, "Initialize K8s submitter failed.");
      }
    }

    coreApi = new CoreV1Api();

    appsV1Api = new AppsV1Api();

    customObjectsApi = new CustomObjectsApi();

    podClient =
            new GenericKubernetesApi<>(
                    V1Pod.class, V1PodList.class,
                    "", "v1", "pods", client);

    eventClient =
            new GenericKubernetesApi<>(
                    CoreV1Event.class, CoreV1EventList.class,
                    "", "v1", "events", client);

    persistentVolumeClaimClient =
            new GenericKubernetesApi<>(
                    V1PersistentVolumeClaim.class, V1PersistentVolumeClaimList.class,
                    "", "v1", "persistentvolumeclaims", client);

    configMapClient =
            new GenericKubernetesApi<>(
                    V1ConfigMap.class, V1ConfigMapList.class,
                    "", "v1", "configmaps", client);

    tfJobClient =
            new GenericKubernetesApi<>(
                    TFJob.class, TFJobList.class,
                    TFJob.CRD_TF_GROUP_V1, TFJob.CRD_TF_VERSION_V1,
                    TFJob.CRD_TF_PLURAL_V1, client);

    pyTorchJobClient =
            new GenericKubernetesApi<>(
                    PyTorchJob.class, PyTorchJobList.class,
                    PyTorchJob.CRD_PYTORCH_GROUP_V1, PyTorchJob.CRD_PYTORCH_VERSION_V1,
                    PyTorchJob.CRD_PYTORCH_PLURAL_V1, client);

    xgboostJobClient =
            new GenericKubernetesApi<>(
                    XGBoostJob.class, XGBoostJobList.class,
                    XGBoostJob.CRD_XGBOOST_GROUP_V1, XGBoostJob.CRD_XGBOOST_VERSION_V1,
                    XGBoostJob.CRD_XGBOOST_PLURAL_V1, client);

    notebookCRClient =
            new GenericKubernetesApi<>(
                    NotebookCR.class, NotebookCRList.class,
                    NotebookCR.CRD_NOTEBOOK_GROUP_V1, NotebookCR.CRD_NOTEBOOK_VERSION_V1,
                    NotebookCR.CRD_NOTEBOOK_PLURAL_V1, client);

    seldonDeploymentClient =
            new GenericKubernetesApi<>(
                    SeldonDeployment.class, SeldonDeploymentList.class,
                    SeldonConstants.GROUP, SeldonConstants.VERSION,
                    SeldonConstants.PLURAL, client);

    istioVirtualServiceClient =
            new GenericKubernetesApi<>(
                    IstioVirtualService.class, IstioVirtualServiceList.class,
                    IstioConstants.GROUP, IstioConstants.VERSION,
                    IstioConstants.PLURAL, client);
  }

  private <T> T checkApi(T api, Class<?> classOfT) {
    if (api == null) {
      throw new SubmarineRuntimeException(String.format("%s api is null, can not get this object!",
          classOfT.getName()));
    } else {
      return api;
    }
  }

  public ApiClient getApiClient() {
    return checkApi(client, ApiClient.class);
  }

  public CustomObjectsApi getCustomObjectsApi() {
    return checkApi(customObjectsApi, CustomObjectsApi.class);
  }

  public CoreV1Api getCoreApi() {
    return checkApi(coreApi, CoreV1Api.class);
  }

  public AppsV1Api getAppsV1Api() {
    return checkApi(appsV1Api, AppsV1Api.class);
  }

  public GenericKubernetesApi<V1Pod, V1PodList> getPodClient() {
    return checkApi(podClient, V1Pod.class);
  }

  public GenericKubernetesApi<CoreV1Event, CoreV1EventList> getEventClient() {
    return checkApi(eventClient, CoreV1Event.class);
  }

  public GenericKubernetesApi<V1PersistentVolumeClaim, V1PersistentVolumeClaimList>
      getPersistentVolumeClaimClient() {
    return checkApi(persistentVolumeClaimClient, V1PersistentVolumeClaim.class);
  }

  public GenericKubernetesApi<V1ConfigMap, V1ConfigMapList> getConfigMapClient() {
    return checkApi(configMapClient, V1ConfigMap.class);
  }

  public GenericKubernetesApi<TFJob, TFJobList> getTfJobClient() {
    return checkApi(tfJobClient, TFJob.class);
  }

  public GenericKubernetesApi<PyTorchJob, PyTorchJobList> getPyTorchJobClient() {
    return checkApi(pyTorchJobClient, PyTorchJob.class);
  }

  public GenericKubernetesApi<XGBoostJob, XGBoostJobList> getXGBoostJobClient() {
    return checkApi(xgboostJobClient, XGBoostJob.class);
  }

  public GenericKubernetesApi<NotebookCR, NotebookCRList> getNotebookCRClient() {
    return checkApi(notebookCRClient, NotebookCR.class);
  }

  public GenericKubernetesApi<SeldonDeployment, SeldonDeploymentList> getSeldonDeploymentClient() {
    return checkApi(seldonDeploymentClient, SeldonDeployment.class);
  }

  public GenericKubernetesApi<IstioVirtualService, IstioVirtualServiceList> getIstioVirtualServiceClient() {
    return checkApi(istioVirtualServiceClient, IstioVirtualService.class);
  }
}
