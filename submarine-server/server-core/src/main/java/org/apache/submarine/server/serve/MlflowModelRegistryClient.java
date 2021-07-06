package org.apache.submarine.server.serve;

import org.apache.http.HttpResponse;
import org.apache.http.client.HttpClient;
import org.apache.http.client.methods.HttpGet;
import org.apache.http.impl.client.HttpClients;

import java.io.IOException;
import java.net.URI;

public class MlflowModelRegistryClient {

  private final HttpClient client = HttpClients.createDefault();

  public boolean checkModelExist(String modelName){
    HttpGet request = new HttpGet();
    request.setHeader("Content-Type", "application/json");
    String base = "http://submarine-mlflow-service:5000/api/" +
        "2.0/preview/mlflow/registered-models/get";
    String query = "?name=" + modelName;
    request.setURI(URI.create(base + query));
    HttpResponse response;
    try {
      response = client.execute(request);
    } catch (IOException e){
      return false;
    }
    int retryLeft = 5;
    while (response.getStatusLine().getStatusCode() == 429 && retryLeft > 0){
      try {
        Thread.sleep(100);
        response = client.execute(request);
      } catch (Exception e){
        return false;
      }
      retryLeft--;
    }
    return (response.getStatusLine().getStatusCode() != 200) ? false : true;
  }
}
