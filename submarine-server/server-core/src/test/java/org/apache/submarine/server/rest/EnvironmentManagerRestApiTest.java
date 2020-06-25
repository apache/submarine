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

package org.apache.submarine.server.rest;

import java.io.IOException;

import javax.ws.rs.core.Response;

import org.apache.commons.httpclient.methods.GetMethod;
import org.apache.submarine.server.AbstractSubmarineServerTest;
import org.apache.submarine.server.api.environment.Environment;
import org.apache.submarine.server.response.JsonResponse;
import org.junit.AfterClass;
import org.junit.Assert;
import org.junit.BeforeClass;
import org.junit.Test;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

@SuppressWarnings("rawtypes")
public class EnvironmentManagerRestApiTest extends AbstractSubmarineServerTest {

  private static String ENV_PATH =
      "/api/" + RestConstants.V1 + "/" + RestConstants.ENVIRONMENTS;
  private static String ENV_NAME = "my-submarine-env";

  @BeforeClass
  public static void startUp() throws Exception {
    AbstractSubmarineServerTest
        .startUp(EnvironmentManagerRestApiTest.class.getSimpleName());
    Assert.assertTrue(checkIfServerIsRunning());
  }

  @AfterClass
  public static void destroy() throws Exception {
    AbstractSubmarineServerTest.shutDown();
  }

  @Test
  public void testCreateEnvironment() throws Exception {
    String body = loadContent("environment/test_env_1.json");
    run(body, "application/json");
    deleteEnvironment();
  }

  @Test
  public void testUpdateEnvironment() throws IOException {

  }

  @Test
  public void testDeleteEnvironment() throws Exception {
    String body = loadContent("environment/test_env_1.json");
    run(body, "application/json");
    deleteEnvironment();
    
    GetMethod getMethod = httpGet(ENV_PATH + "/" + ENV_NAME);
    Assert.assertEquals(Response.Status.NOT_FOUND.getStatusCode(),
        getMethod.getStatusCode());
  }

  @Test
  public void testGetEnvironment() throws Exception {

    String body = loadContent("environment/test_env_1.json");
    run(body, "application/json");

    Gson gson = new GsonBuilder().create();
    GetMethod getMethod = httpGet(ENV_PATH + "/" + ENV_NAME);
    Assert.assertEquals(Response.Status.OK.getStatusCode(),
        getMethod.getStatusCode());

    String json = getMethod.getResponseBodyAsString();
    JsonResponse jsonResponse = gson.fromJson(json, JsonResponse.class);
    Assert.assertEquals(Response.Status.OK.getStatusCode(),
        jsonResponse.getCode());

    Environment getEnvironment =
        gson.fromJson(gson.toJson(jsonResponse.getResult()), Environment.class);
    Assert.assertEquals(ENV_NAME, getEnvironment.getEnvironmentSpec().getName());
    
    deleteEnvironment();
  }

  @Test
  public void testNotFoundEnvironment() throws Exception {

    Gson gson = new GsonBuilder().create();

    GetMethod getMethod = httpGet(ENV_PATH + "/" + "no_such_env_exists");
    Assert.assertEquals(Response.Status.NOT_FOUND.getStatusCode(),
        getMethod.getStatusCode());

    String json = getMethod.getResponseBodyAsString();
    JsonResponse jsonResponse = gson.fromJson(json, JsonResponse.class);
    Assert.assertEquals(Response.Status.NOT_FOUND.getStatusCode(),
        jsonResponse.getCode());
  }
}
