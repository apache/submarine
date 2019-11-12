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

import com.google.gson.Gson;
import org.apache.submarine.jobserver.rest.dao.JsonResponse;
import org.apache.submarine.jobserver.rest.dao.RestConstants;
import org.apache.submarine.jobserver.rest.api.JobApi;
import org.glassfish.jersey.server.ResourceConfig;
import org.glassfish.jersey.test.JerseyTest;
import org.junit.Test;

import javax.ws.rs.client.Entity;
import javax.ws.rs.core.Application;
import javax.ws.rs.core.MediaType;
import javax.ws.rs.core.Response;

import static org.junit.Assert.assertEquals;

public class JobApiTest extends JerseyTest {

  @Override
  protected Application configure() {
    return new ResourceConfig(JobApi.class);
  }

  @Test
  public void testJobServerPing() {
    String str = "Pong";
    Response response = target(RestConstants.V1 + "/"
        + RestConstants.JOBS + "/" + RestConstants.PING)
        .request()
        .get();
    Gson gson = new Gson();
    JsonResponse r = gson.fromJson(response.readEntity(String.class),
        JsonResponse.class);
    assertEquals("Response code should be 200 ",
        Response.Status.OK.getStatusCode(), response.getStatus());
    assertEquals("Response message should be " + str,
        str, r.getResult().toString());
  }

  // Test job created with correct JSON input
  @Test
  public void testCreateJobWhenJsonInputIsCorrectThenResponseCodeAccepted() {
    String jobSpec = "{\"type\": \"tensorflow\", \"version\":\"v1.13\"}";
    Response response = target(RestConstants.V1 + "/" + RestConstants.JOBS)
        .request()
        .post(Entity.entity(jobSpec, MediaType.APPLICATION_JSON));

    assertEquals("Http Response should be 202 ",
        Response.Status.ACCEPTED.getStatusCode(), response.getStatus());
  }

  // Test job created with incorrect JSON input
  @Test
  public void testCreateJobWhenJsonInputIsWrongThenResponseCodeBadRequest() {
    String jobSpec = "{\"ttype\": \"tensorflow\", \"version\":\"v1.13\"}";
    Response response = target(RestConstants.V1 + "/" + RestConstants.JOBS)
        .request()
        .post(Entity.entity(jobSpec, MediaType.APPLICATION_JSON));

    assertEquals("Http Response should be 400 ",
        Response.Status.BAD_REQUEST.getStatusCode(), response.getStatus());
  }

  // Test get job list
  @Test
  public void testGetJobList() {
    Response response = target(RestConstants.V1 + "/" + RestConstants.JOBS)
        .request()
        .get();

    assertEquals("Http Response should be 200 ",
        Response.Status.OK.getStatusCode(), response.getStatus());
  }

  // Test get job by id
  @Test
  public void testGetJobById() {
    String jobId = "job1";
    Response response = target(RestConstants.V1 + "/"
        + RestConstants.JOBS + "/" + jobId)
        .request()
        .get();
    Gson gson = new Gson();
    JsonResponse r = gson.fromJson(response.readEntity(String.class),
        JsonResponse.class);
    assertEquals("Http Response should be 200 ",
        Response.Status.OK.getStatusCode(), response.getStatus());
    assertEquals("Job id should be " + jobId,
        jobId, r.getResult().toString());
  }

  // Test delete job by id
  @Test
  public void testDeleteJobById() {
    String jobId = "job1";
    Response response = target(RestConstants.V1 + "/"
        + RestConstants.JOBS + "/" + jobId)
        .request()
        .delete();
    Gson gson = new Gson();
    JsonResponse r = gson.fromJson(response.readEntity(String.class),
        JsonResponse.class);
    assertEquals("Http Response should be 200 ",
        Response.Status.OK.getStatusCode(), response.getStatus());
    assertEquals("Deleted job id should be " + jobId,
        jobId, r.getResult().toString());
  }

  /**
   * FiXME. The manual YAML test with postman works but failed here.
   * We need to figure out why the YAML entity provider not work in this test.
   * */
  @Test
  public void testCreateJobWhenYamlInputIsCorrectThenResponseCodeAccepted() {
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
