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
import io.kubernetes.client.common.KubernetesObject;
import io.kubernetes.client.openapi.models.V1ObjectMeta;
import org.apache.submarine.serve.utils.IstioConstants;
import org.apache.submarine.server.k8s.utils.K8sUtils;

public class IstioVirtualService implements KubernetesObject {
  @SerializedName("apiVersion")
  private String apiVersion = IstioConstants.API_VERSION;

  @SerializedName("kind")
  private String kind = IstioConstants.KIND;

  @SerializedName("metadata")
  private V1ObjectMeta metadata;

  @SerializedName("spec")
  private IstioVirtualServiceSpec spec;

  // transient to avoid being serialized
  private transient String group = IstioConstants.GROUP;

  private transient String version = IstioConstants.VERSION;

  private transient String plural = IstioConstants.PLURAL;

  public IstioVirtualService(V1ObjectMeta metadata, IstioVirtualServiceSpec spec) {
    this.metadata = metadata;
    this.spec = spec;
  }

  public IstioVirtualService(Long id, String modelResourceName, Integer modelVersion) {
    V1ObjectMeta metadata = new V1ObjectMeta();
    metadata.setName(modelResourceName);
    metadata.setNamespace(K8sUtils.getNamespace());
    setMetadata(metadata);
    setSpec(new IstioVirtualServiceSpec(id, modelResourceName, modelVersion));
  }

  public IstioVirtualService(V1ObjectMeta metadata) {
    this.metadata = metadata;
    this.spec = parseVirtualServiceSpec(metadata.getNamespace(), metadata.getName());
  }

  protected IstioVirtualServiceSpec parseVirtualServiceSpec(String namespace, String name) {
    IstioVirtualServiceSpec spec = new IstioVirtualServiceSpec();
    spec.addHost(IstioConstants.DEFAULT_INGRESS_HOST);
    spec.addGateway(IstioConstants.SUBMARINE_GATEWAY);
    String matchURIPrefix = "/notebook/" + namespace + "/" + name;
    spec.setHTTPRoute(new IstioHTTPRoute(matchURIPrefix, name, 80));
    return spec;
  }

  public String getApiVersion() {
    return apiVersion;
  }

  public void setApiVersion(String apiVersion) {
    this.apiVersion = apiVersion;
  }

  public String getKind() {
    return kind;
  }

  public void setKind(String kind) {
    this.kind = kind;
  }

  public V1ObjectMeta getMetadata() {
    return metadata;
  }

  public void setMetadata(V1ObjectMeta metadata) {
    this.metadata = metadata;
  }

  public String getGroup() {
    return group;
  }

  public void setGroup(String group) {
    this.group = group;
  }

  public String getVersion() {
    return version;
  }

  public void setVersion(String version) {
    this.version = version;
  }

  public String getPlural() {
    return plural;
  }

  public void setPlural(String plural) {
    this.plural = plural;
  }

  public IstioVirtualServiceSpec getSpec() {
    return spec;
  }

  public void setSpec(IstioVirtualServiceSpec istioVirtualServiceSpec){
    this.spec = istioVirtualServiceSpec;
  }

}
