package org.apache.submarine.serve.istio;

import com.google.gson.annotations.SerializedName;
import org.apache.submarine.serve.utils.IstioConstants;

import java.util.ArrayList;
import java.util.List;

public class IstioVirtualServiceSpec {
  @SerializedName("hosts")
  private List<String> hosts = new ArrayList<>();
  @SerializedName("gateways")
  private List<String> gateways = new ArrayList<>();
  @SerializedName("http")
  private List<IstioHTTPRoute> httpRoute = new ArrayList<>();

  public IstioVirtualServiceSpec() {
  }

  public IstioVirtualServiceSpec(String modelName, Integer modelVersion) {
    hosts.add(IstioConstants.DEFAULT_INGRESS_HOST);
    gateways.add(IstioConstants.DEFAULT_GATEWAY);
    IstioHTTPDestination destination = new IstioHTTPDestination(
            modelName + "-" + IstioConstants.DEFAULT_NAMESPACE);
    IstioHTTPMatchRequest matchRequest = new IstioHTTPMatchRequest("/" + modelName
            + "/" + String.valueOf(modelVersion) + "/");
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
}
