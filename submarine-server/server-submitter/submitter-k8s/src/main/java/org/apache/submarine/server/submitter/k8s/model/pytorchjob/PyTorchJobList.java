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

package org.apache.submarine.server.submitter.k8s.model.pytorchjob;

import com.google.gson.annotations.SerializedName;
import java.util.List;

import io.kubernetes.client.common.KubernetesListObject;
import io.kubernetes.client.common.KubernetesObject;
import io.kubernetes.client.openapi.models.V1ListMeta;

public class PyTorchJobList implements KubernetesListObject{

  @SerializedName("apiVersion")
  private String apiVersion;

  @SerializedName("kind")
  private String kind;

  @SerializedName("metadata")
  private V1ListMeta metadata;

  @SerializedName("items")
  private List<PyTorchJob> items;

  @Override
  public V1ListMeta getMetadata() {
    return metadata;
  }

  @Override
  public List<? extends KubernetesObject> getItems() {
    return items;
  }

  @Override
  public String getApiVersion() {
    return PyTorchJob.CRD_PYTORCH_API_VERSION_V1;
  }

  @Override
  public String getKind() {
    return PyTorchJob.CRD_PYTORCH_KIND_V1 + "List";
  }
}
