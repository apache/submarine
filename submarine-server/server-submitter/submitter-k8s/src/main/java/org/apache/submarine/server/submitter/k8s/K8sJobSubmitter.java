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

import com.google.common.annotations.VisibleForTesting;
import com.google.gson.Gson;
import io.kubernetes.client.ApiClient;
import io.kubernetes.client.ApiException;
import io.kubernetes.client.Configuration;
import io.kubernetes.client.apis.CustomObjectsApi;
import io.kubernetes.client.models.V1DeleteOptions;
import io.kubernetes.client.models.V1DeleteOptionsBuilder;
import io.kubernetes.client.util.ClientBuilder;
import io.kubernetes.client.util.KubeConfig;
import org.apache.submarine.server.submitter.k8s.model.CustomResourceJob;
import org.apache.submarine.server.submitter.k8s.model.CustomResourceJobList;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.FileReader;
import java.io.IOException;

/**
 * JobSubmitter for Kubernetes Cluster.
 */
// TODO: It should implement the JobSubmitter interface
public class K8sJobSubmitter {
  private final Logger LOG = LoggerFactory.getLogger(K8sJobSubmitter.class);

  public K8sJobSubmitter(String confPath) throws IOException {
    ApiClient client =
        ClientBuilder.kubeconfig(KubeConfig.loadKubeConfig(new FileReader(confPath))).build();
    Configuration.setDefaultApiClient(client);
  }

  public String submitJob(K8sJobRequest request) {
    return "job_id";
  }

  @VisibleForTesting
  CustomResourceJob createCustomJob(K8sJobRequest request) {
    try {
      CustomObjectsApi api = new CustomObjectsApi();
      K8sJobRequest.Path path = request.getPath();
      Object o = api.createNamespacedCustomObject(path.getGroup(),
          path.getApiVersion(), path.getNamespace(), path.getPlural(),
          request.getBody(), "true");
      Gson gson = new Gson();
      return gson.fromJson(gson.toJson(o), CustomResourceJob.class);
    } catch (ApiException ae) {
      LOG.error("Create CRD job: " + ae.getMessage(), ae);
    }
    return null;
  }

  @VisibleForTesting
  CustomResourceJob getCustomResourceJob(K8sJobRequest request) {
    try {
      CustomObjectsApi api = new CustomObjectsApi();
      K8sJobRequest.Path path = request.getPath();
      Object o = api.getNamespacedCustomObject(path.getGroup(), path.getApiVersion(),
          path.getNamespace(), path.getPlural(), request.getJobName());
      Gson gson = new Gson();
      return gson.fromJson(gson.toJson(o), CustomResourceJob.class);
    } catch (ApiException ae) {
      LOG.error("Get CRD job: " + ae.getMessage(), ae);
    }
    return null;
  }

  CustomResourceJob deleteCustomResourceJob(K8sJobRequest request) {
    try {
      CustomObjectsApi api = new CustomObjectsApi();
      K8sJobRequest.Path path = request.getPath();
      V1DeleteOptions body =
          new V1DeleteOptionsBuilder().withApiVersion(path.getApiVersion()).build();
      Object o = api.deleteNamespacedCustomObject(path.getGroup(),
          path.getApiVersion(), path.getNamespace(), path.getPlural(),
          request.getJobName(), body, null, null, null);
      Gson gson = new Gson();
      return gson.fromJson(gson.toJson(o), CustomResourceJob.class);
    } catch (ApiException ae) {
      LOG.error("Delete CRD job: " + ae.getMessage(), ae);
    }
    return null;
  }

  @VisibleForTesting
  CustomResourceJobList listCustomResourceJobs(K8sJobRequest request) {
    try {
      CustomObjectsApi api = new CustomObjectsApi();
      K8sJobRequest.Path path = request.getPath();
      Object o = api.listNamespacedCustomObject(path.getGroup(), path.getApiVersion(),
          path.getNamespace(), path.getPlural(), "true", null, null, null, null, null);
      Gson gson = new Gson();
      return gson.fromJson(gson.toJson(o), CustomResourceJobList.class);
    } catch (ApiException ae) {
      LOG.error("List CRD jobs: " + ae.getMessage(), ae);
    }
    return null;
  }
}
