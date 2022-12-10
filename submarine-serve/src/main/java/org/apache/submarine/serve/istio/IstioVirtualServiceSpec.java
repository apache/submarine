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

import com.fasterxml.jackson.annotation.JsonProperty;
import com.google.gson.annotations.SerializedName;
import org.apache.submarine.serve.utils.IstioConstants;
import org.apache.submarine.server.k8s.utils.K8sUtils;

import java.util.ArrayList;
import java.util.List;

public class IstioVirtualServiceSpec {

  @SerializedName("hosts")
  private List<String> hosts = new ArrayList<>();
  @SerializedName("gateways")
  private List<String> gateways = new ArrayList<>();
  @SerializedName("http")
  @JsonProperty("http")
  private List<IstioHTTPRoute> httpRoute = new ArrayList<>();

  public IstioVirtualServiceSpec() {
  }

  public IstioVirtualServiceSpec(Long id, String modelResourceName, Integer modelVersion) {
    hosts.add(IstioConstants.DEFAULT_INGRESS_HOST);
    gateways.add(IstioConstants.SELDON_GATEWAY);
    // model resource name is service name
    IstioHTTPDestination destination = new IstioHTTPDestination(modelResourceName);
    IstioHTTPMatchRequest matchRequest = new IstioHTTPMatchRequest(
        // use namespace to avoid multi tenant problems
        // use model_version_id replaced model_name to avoid illegal characters and other problems
        // use model_version as the identification of the model
        String.format("/seldon/%s/%s/%s/", K8sUtils.getNamespace(), id, modelVersion)
    );
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

  public void setHTTPRoute(IstioHTTPRoute istioHTTPRoute) {
    this.httpRoute.add(istioHTTPRoute);
  }

  public void setHosts(List<String> hosts) {
    this.hosts = hosts;
  }

  public void setGateways(List<String> gateways) {
    this.gateways = gateways;
  }

  public List<IstioHTTPRoute> getHttpRoute() {
    return httpRoute;
  }

  public void setHttpRoute(List<IstioHTTPRoute> httpRoute) {
    this.httpRoute = httpRoute;
  }
}
