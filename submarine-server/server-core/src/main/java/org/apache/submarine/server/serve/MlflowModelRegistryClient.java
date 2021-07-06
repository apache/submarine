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
