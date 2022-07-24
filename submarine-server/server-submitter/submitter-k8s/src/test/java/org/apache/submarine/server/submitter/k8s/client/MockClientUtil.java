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

import io.kubernetes.client.openapi.models.V1Status;
import io.kubernetes.client.openapi.models.V1StatusDetails;
import org.apache.submarine.server.submitter.k8s.util.JsonUtils;

public class MockClientUtil {

  public static String getNotebookUrl(String namespace, String name) {
    return String.format("/apis/kubeflow.org/v1/namespaces/%s/notebooks/%s", namespace, name);
  }

  public static String getIstioUrl(String namespace, String name) {
    return String.format("/apis/networking.istio.io/v1beta1/namespaces/%s/virtualservices/%s",
            namespace, name);
  }

  public static String getPvcUrl(String namespace, String name) {
    return String.format("/api/v1/namespaces/%s/persistentvolumeclaims/%s", namespace, name);
  }

  public static String getConfigmapUrl(String namespace, String name) {
    return String.format("/api/v1/namespaces/%s/configmaps/%s", namespace, name);
  }

  public static String getPodUrl(String namespace, String name) {
    return String.format("/api/v1/namespaces/%s/pods/%s", namespace, name);
  }

  public static String getTfJobUrl(String namespace, String name) {
    return String.format("/apis/kubeflow.org/v1/namespaces/%s/tfjobs/%s", namespace, name);
  }

  public static String getPytorchJobUrl(String namespace, String name) {
    return String.format("/apis/kubeflow.org/v1/namespaces/%s/pytorchjobs/%s", namespace, name);
  }

  public static String getMockSuccessStatus(String name) {
    V1Status status = new V1Status();
    status.setApiVersion("v1");
    status.setKind("Status");
    status.setStatus("Success");
    V1StatusDetails details = new V1StatusDetails();
    details.setName(name);
    return JsonUtils.toJson(status);
  }

  public static String getMockSuccessStatus(String group, String kind, String name) {
    V1Status status = new V1Status();
    status.setApiVersion("v1");
    status.setKind("Status");
    status.setStatus("Success");
    V1StatusDetails details = new V1StatusDetails();
    details.setGroup(group);
    details.setKind(kind);
    details.setName(name);
    return JsonUtils.toJson(status);
  }

}
