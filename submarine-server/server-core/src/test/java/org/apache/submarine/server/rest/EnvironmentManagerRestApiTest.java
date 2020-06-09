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

import java.io.File;
import java.io.IOException;
import java.io.Reader;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;

import javax.ws.rs.core.Response;

import org.apache.commons.httpclient.methods.DeleteMethod;
import org.apache.commons.httpclient.methods.GetMethod;
import org.apache.commons.httpclient.methods.PostMethod;
import org.apache.commons.io.FileUtils;
import org.apache.submarine.server.AbstractSubmarineServerTest;
import org.apache.submarine.server.api.environment.Environment;
import org.apache.submarine.server.api.spec.EnvironmentSpec;
import org.apache.submarine.server.api.spec.ExperimentSpec;
import org.apache.submarine.server.response.JsonResponse;
import org.junit.AfterClass;
import org.junit.Assert;
import org.junit.BeforeClass;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

@SuppressWarnings("rawtypes")
public class EnvironmentManagerRestApiTest extends AbstractSubmarineServerTest {
  private static final Logger LOG =
      LoggerFactory.getLogger(EnvironmentManagerRestApiTest.class);

  private static String ENV_PATH =
      "/api/" + RestConstants.V1 + "/" + RestConstants.ENVIRONMENTS;

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
  }

  @Test
  public void testUpdateEnvironment() throws IOException {

  }

  @Test
  public void testDeleteEnvironment() throws Exception {

    String body = loadContent("environment/test_env_1.json");
    run(body, "application/json");

    Gson gson = new GsonBuilder().create();
    String envName = "my_submarine_env";
    DeleteMethod deleteMethod = httpDelete(ENV_PATH + "/" + envName);
    Assert.assertEquals(Response.Status.OK.getStatusCode(),
        deleteMethod.getStatusCode());

    String json = deleteMethod.getResponseBodyAsString();
    JsonResponse jsonResponse = gson.fromJson(json, JsonResponse.class);
    Assert.assertEquals(Response.Status.OK.getStatusCode(),
        jsonResponse.getCode());

    Environment deletedEnv =
        gson.fromJson(gson.toJson(jsonResponse.getResult()), Environment.class);

    Assert.assertEquals(envName, deletedEnv.getName());

  }

  @Test
  public void testGetEnvironment() throws Exception {

    String body = loadContent("environment/test_env_1.json");
    run(body, "application/json");

    Gson gson = new GsonBuilder().create();
    String envName = "my_submarine_env";
    GetMethod getMethod = httpGet(ENV_PATH + "/" + envName);
    Assert.assertEquals(Response.Status.OK.getStatusCode(),
        getMethod.getStatusCode());

    String json = getMethod.getResponseBodyAsString();
    JsonResponse jsonResponse = gson.fromJson(json, JsonResponse.class);
    Assert.assertEquals(Response.Status.OK.getStatusCode(),
        jsonResponse.getCode());

    Environment getEnvironment =
        gson.fromJson(gson.toJson(jsonResponse.getResult()), Environment.class);
    Assert.assertEquals(envName, getEnvironment.getName());
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

  private void run(String body, String contentType) throws Exception {

    Gson gson = new GsonBuilder().create();

    // create
    LOG.info("Create Environment using Environment REST API");

    PostMethod postMethod = httpPost(ENV_PATH, body, contentType);
    Assert.assertEquals(Response.Status.OK.getStatusCode(),
        postMethod.getStatusCode());

    String json = postMethod.getResponseBodyAsString();
    JsonResponse jsonResponse = gson.fromJson(json, JsonResponse.class);
    Assert.assertEquals(Response.Status.OK.getStatusCode(),
        jsonResponse.getCode());

    Environment env =
        gson.fromJson(gson.toJson(jsonResponse.getResult()), Environment.class);
    verifyCreateEnvironmentApiResult(env);

  }

  private void verifyCreateEnvironmentApiResult(Environment env)
      throws Exception {
    Assert.assertNotNull(env.getName());
    Assert.assertNotNull(env.getEnvironmentSpec());
  }

  String loadContent(String resourceName) throws Exception {
    URL fileUrl = this.getClass().getResource("/" + resourceName);
    LOG.info("Resource file: " + fileUrl);
    return FileUtils.readFileToString(new File(fileUrl.toURI()),
        StandardCharsets.UTF_8);
  }

  protected ExperimentSpec buildFromJsonFile(String filePath)
      throws IOException, URISyntaxException {
    Gson gson = new GsonBuilder().create();
    try (Reader reader = Files.newBufferedReader(
        getCustomJobSpecFile(filePath).toPath(), StandardCharsets.UTF_8)) {
      return (ExperimentSpec) gson.fromJson(reader, ExperimentSpec.class);
    }
  }

  private File getCustomJobSpecFile(String path) throws URISyntaxException {
    URL fileUrl = this.getClass().getResource(path);
    return new File(fileUrl.toURI());
  }

}