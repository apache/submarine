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

  public boolean checkModelExist(String modelName, String modelVersion){
    HttpClient client = HttpClients.createDefault();
    HttpGet request = new HttpGet();
    request.setHeader("Content-Type", "application/json");
    String base = "http://submarine-mlflow-service:5000/api/" +
        "2.0/preview/mlflow/model-versions/get";
    String query = "?name=" + modelName + "&version=" + modelVersion;
    request.setURI(URI.create(base + query));
    HttpResponse response;
    try {
      response = client.execute(request);
    } catch (IOException e){
      return false;
    }
    return response.getStatusLine().getStatusCode() == 200;
  }
}
