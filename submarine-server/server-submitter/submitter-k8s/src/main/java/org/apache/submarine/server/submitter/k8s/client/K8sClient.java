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
import io.kubernetes.client.util.generic.GenericKubernetesApi;
import org.apache.submarine.serve.istio.IstioVirtualService;
import org.apache.submarine.serve.istio.IstioVirtualServiceList;
import org.apache.submarine.serve.seldon.SeldonDeployment;
import org.apache.submarine.serve.seldon.SeldonDeploymentList;
import org.apache.submarine.server.submitter.k8s.model.notebook.NotebookCR;
import org.apache.submarine.server.submitter.k8s.model.notebook.NotebookCRList;
import org.apache.submarine.server.submitter.k8s.model.pytorchjob.PyTorchJob;
import org.apache.submarine.server.submitter.k8s.model.pytorchjob.PyTorchJobList;
import org.apache.submarine.server.submitter.k8s.model.tfjob.TFJob;
import org.apache.submarine.server.submitter.k8s.model.tfjob.TFJobList;
import org.apache.submarine.server.submitter.k8s.model.xgboostjob.XGBoostJob;
import org.apache.submarine.server.submitter.k8s.model.xgboostjob.XGBoostJobList;

public interface K8sClient {

  ApiClient getApiClient();

  CustomObjectsApi getCustomObjectsApi();

  CoreV1Api getCoreApi();

  AppsV1Api getAppsV1Api();

  GenericKubernetesApi<V1Pod, V1PodList> getPodClient();

  GenericKubernetesApi<CoreV1Event, CoreV1EventList> getEventClient();

  GenericKubernetesApi<V1PersistentVolumeClaim, V1PersistentVolumeClaimList>
      getPersistentVolumeClaimClient();

  GenericKubernetesApi<V1ConfigMap, V1ConfigMapList> getConfigMapClient();

  GenericKubernetesApi<TFJob, TFJobList> getTfJobClient();

  GenericKubernetesApi<PyTorchJob, PyTorchJobList> getPyTorchJobClient();

  GenericKubernetesApi<XGBoostJob, XGBoostJobList> getXGBoostJobClient();

  GenericKubernetesApi<NotebookCR, NotebookCRList> getNotebookCRClient();

  GenericKubernetesApi<SeldonDeployment, SeldonDeploymentList> getSeldonDeploymentClient();

  GenericKubernetesApi<IstioVirtualService, IstioVirtualServiceList> getIstioVirtualServiceClient();

}
