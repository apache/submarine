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

package org.apache.submarine.serve.istio;

import com.google.gson.annotations.SerializedName;
import java.util.List;

import io.kubernetes.client.common.KubernetesListObject;
import io.kubernetes.client.common.KubernetesObject;
import io.kubernetes.client.openapi.models.V1ListMeta;
import org.apache.submarine.serve.utils.IstioConstants;

public class IstioVirtualServiceList implements KubernetesListObject{

  @SerializedName("apiVersion")
  private String apiVersion;

  @SerializedName("kind")
  private String kind;

  @SerializedName("metadata")
  private V1ListMeta metadata;

  @SerializedName("items")
  private List<IstioVirtualService> items;

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
    return IstioConstants.API_VERSION;
  }

  @Override
  public String getKind() {
    return IstioConstants.KIND + "List";
  }
}
