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

package org.apache.submarine.server.submitter.k8s.model.notebook;

import com.google.gson.annotations.SerializedName;

import io.kubernetes.client.common.KubernetesListObject;
import io.kubernetes.client.openapi.models.V1ListMeta;

import java.util.List;

public class NotebookCRList implements KubernetesListObject{

  public static final String CRD_NOTEBOOK_VERSION_V1 = "v1alpha1";
  public static final String CRD_NOTEBOOK_GROUP_V1 = "kubeflow.org";
  public static final String CRD_APIVERSION_V1 = CRD_NOTEBOOK_GROUP_V1 + "/" + CRD_NOTEBOOK_VERSION_V1;
  public static final String CRD_NOTEBOOK_LIST_KIND_V1 = "NotebookList";

  public NotebookCRList() {
    setApiVersion(CRD_APIVERSION_V1);
    setKind(CRD_NOTEBOOK_LIST_KIND_V1);
  }
  
  @SerializedName("apiVersion")
  private String apiVersion;

  @SerializedName("kind")
  private String kind;

  @SerializedName("metadata")
  private V1ListMeta metadata;
    
  @SerializedName("items")
  private List<NotebookCR> items;
  
  public void setApiVersion(String apiVersion) {
    this.apiVersion = apiVersion;
  }

  public void setKind(String kind) {
    this.kind = kind;
  }

  @Override
  public V1ListMeta getMetadata() {
    return metadata;
  }

  @Override
  public List<NotebookCR> getItems() {
    return items;
  }

  @Override
  public String getApiVersion() {
    return apiVersion;
  }

  @Override
  public String getKind() {
  
    return kind;
  }
}

