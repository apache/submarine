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

package org.apache.submarine.server.jobserver;

import com.google.gson.Gson;
import org.apache.commons.httpclient.methods.DeleteMethod;
import org.apache.commons.httpclient.methods.GetMethod;
import org.apache.commons.httpclient.methods.PostMethod;
import org.apache.submarine.server.AbstractSubmarineServerTest;
import org.apache.submarine.server.rest.RestConstants;
import org.apache.submarine.server.response.JsonResponse;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.ws.rs.core.Response;
import java.io.IOException;

import static org.junit.Assert.assertEquals;

public class JobServerRestApiTest extends AbstractSubmarineServerTest {
  private static final Logger LOG = LoggerFactory.getLogger(JobServerRestApiTest.class);

  @BeforeClass
  public static void init() throws Exception {
    AbstractSubmarineServerTest.startUp(JobServerRestApiTest.class.getSimpleName());
  }

  @AfterClass
  public static void destroy() throws Exception {
    AbstractSubmarineServerTest.shutDown();
  }

  @Test
  public void testJobServerPing() throws IOException {
    GetMethod response = httpGet("/api/" + RestConstants.V1 + "/"
        + RestConstants.JOBS + "/" + RestConstants.PING);
    LOG.info(response.toString());

    String requestBody = response.getResponseBodyAsString();
    LOG.info(requestBody);

    Gson gson = new Gson();
    JsonResponse jsonResponse = gson.fromJson(requestBody, JsonResponse.class);
    assertEquals("Response code should be 200 ",
        Response.Status.OK.getStatusCode(), jsonResponse.getCode());
    assertEquals("Response result should be Pong", "Pong",
        jsonResponse.getResult().toString());
  }

  // Test job created with correct JSON input
  @Test
  public void testCreateJobWhenJsonInputIsCorrectThenResponseCodeAccepted() throws IOException {
    String jobSpec = "{\"name\": \"mnist\"}";

    PostMethod response = httpPost("/api/" + RestConstants.V1 + "/" + RestConstants.JOBS, jobSpec);
    LOG.info(response.toString());

    String requestBody = response.getResponseBodyAsString();
    LOG.info(requestBody);

    Gson gson = new Gson();
    JsonResponse jsonResponse = gson.fromJson(requestBody, JsonResponse.class);
    assertEquals("Response code should be 202 ",
        Response.Status.ACCEPTED.getStatusCode(), jsonResponse.getCode());
  }

  // Test job created with incorrect JSON input
  @Test
  public void testCreateJobWhenJsonInputIsWrongThenResponseCodeBadRequest() throws IOException {
    String jobSpec = "{\"ttype\": \"tensorflow\", \"version\":\"v1.13\"}";

    PostMethod response = httpPost("/api/" + RestConstants.V1 + "/" + RestConstants.JOBS, jobSpec);

    assertEquals("Http Response should be 400 ",
        Response.Status.BAD_REQUEST.getStatusCode(), response.getStatusCode());
  }

  // Test get job list
  @Test
  public void testGetJobList() throws IOException {
    GetMethod response = httpGet("/api/" + RestConstants.V1 + "/" + RestConstants.JOBS);
    LOG.info(response.toString());

    String requestBody = response.getResponseBodyAsString();
    LOG.info(requestBody);

    Gson gson = new Gson();
    JsonResponse jsonResponse = gson.fromJson(requestBody, JsonResponse.class);
    assertEquals("Response code should be 200 ",
        Response.Status.OK.getStatusCode(), jsonResponse.getCode());
  }

  // Test get job by id
  @Test
  public void testGetJobById() throws IOException {
    String jobId = "job1";
    GetMethod response = httpGet("/api/" + RestConstants.V1 + "/"
        + RestConstants.JOBS + "/" + jobId);
    LOG.info(response.toString());

    String requestBody = response.getResponseBodyAsString();
    LOG.info(requestBody);

    Gson gson = new Gson();
    JsonResponse jsonResponse = gson.fromJson(requestBody, JsonResponse.class);
    assertEquals("Response code should be 200 ",
        Response.Status.OK.getStatusCode(), jsonResponse.getCode());
    assertEquals("Job id should be " + jobId, jobId,
        jsonResponse.getResult().toString());
  }

  // Test delete job by id
  @Test
  public void testDeleteJobById() throws IOException {
    String jobId = "job1";
    DeleteMethod response = httpDelete("/api/" + RestConstants.V1 + "/"
        + RestConstants.JOBS + "/" + jobId);
    LOG.info(response.toString());

    String requestBody = response.getResponseBodyAsString();
    LOG.info(requestBody);

    Gson gson = new Gson();
    JsonResponse jsonResponse = gson.fromJson(requestBody, JsonResponse.class);
    assertEquals("Response code should be 200 ",
        Response.Status.OK.getStatusCode(), jsonResponse.getCode());
  }

  /**
   * FiXME. The manual YAML test with postman works but failed here.
   * We need to figure out why the YAML entity provider not work in this test.
   * */
  @Test
  public void testCreateJobWhenYamlInputIsCorrectThenResponseCodeAccepted() throws IOException {
//    Client client = ClientBuilder.newBuilder()
//        .register(new YamlEntityProvider<>()).build();
//    this.setClient(client);
//    String jobSpec = "type: tf";
//    Response response = target(RestConstants.V1 + "/"
//        + RestConstants.JOBS + "/" + "test")
//        .request()
//        .put(Entity.entity(jobSpec, "application/yaml"));
//
//    assertEquals("Http Response should be 202 ",
//        Response.Status.ACCEPTED.getStatusCode(), response.getStatus());
  }
}
