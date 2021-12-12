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
import io.kubernetes.client.models.V1ObjectMeta;
import org.apache.submarine.serve.utils.IstioConstants;

import java.util.ArrayList;
import java.util.List;

public class IstioVirtualService {
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

  public IstioVirtualService(String modelName, Integer modelVersion) {
    V1ObjectMeta metadata = new V1ObjectMeta();
    metadata.setName(modelName);
    metadata.setNamespace(IstioConstants.DEFAULT_NAMESPACE);
    setMetadata(metadata);
    setSpec(new IstioVirtualServiceSpec(modelName, modelVersion));
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

  public static class IstioVirtualServiceSpec {
    @SerializedName("hosts")
    private List<String> hosts = new ArrayList<>();
    @SerializedName("gateways")
    private List<String> gateways = new ArrayList<>();
    @SerializedName("http")
    private List<IstioHTTPRoute> httpRoute = new ArrayList<>();

    public IstioVirtualServiceSpec(String modelName, Integer modelVersion) {
      hosts.add(IstioConstants.DEFAULT_INGRESS_HOST);
      gateways.add(IstioConstants.DEFAULT_GATEWAY);
      IstioHTTPDestination destination = new IstioHTTPDestination(
          modelName + "-" + IstioConstants.DEFAULT_NAMESPACE);
      IstioHTTPMatchRequest matchRequest = new IstioHTTPMatchRequest("/" + modelName
          + "/"  + String.valueOf(modelVersion) + "/");
      IstioHTTPRoute httpRoute = new IstioHTTPRoute();
      httpRoute.addHTTPDestination(destination);
      httpRoute.addHTTPMatchRequest(matchRequest);
      setHTTPRoute(httpRoute);
    }

    public List<String> getHosts() {
      return this.hosts;
    }

    public void addHost(String host) {
      hosts.add(host);
    }

    public List<String> getGateways() {
      return this.gateways;
    }

    public void addGateway(String gateway) {
      gateways.add(gateway);
    }

    public void setHTTPRoute(IstioHTTPRoute istioHTTPRoute){
      this.httpRoute.add(istioHTTPRoute);
    }
  }
}
