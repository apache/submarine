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

package org.apache.submarine.rest;

import java.io.IOException;

import javax.ws.rs.core.Response;

import org.apache.commons.httpclient.methods.DeleteMethod;
import org.apache.commons.httpclient.methods.GetMethod;
import org.apache.commons.httpclient.methods.PostMethod;
import org.apache.submarine.server.AbstractSubmarineServerTest;
import org.apache.submarine.server.api.experimenttemplate.ExperimentTemplate;
import org.apache.submarine.server.response.JsonResponse;
import org.apache.submarine.server.rest.RestConstants;
import org.junit.Assert;
import org.junit.BeforeClass;
import org.junit.Test;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

@SuppressWarnings("rawtypes")
public class ExperimentTemplateManagerRestApiIT extends AbstractSubmarineServerTest {

  protected static String TPL_PATH =
      "/api/" + RestConstants.V1 + "/" + RestConstants.EXPERIMENT_TEMPLATES;
  protected static String TPL_NAME = "tf-mnist-test2";


  @BeforeClass
  public static void startUp() throws Exception {
    Assert.assertTrue(checkIfServerIsRunning());
  }

  @Test
  public void testCreateExperimentTemplate() throws Exception {
    String body = loadContent("experimenttemplate/test_template_2.json");
    run(body, "application/json");
    deleteExperimentTemplate();
  }

  @Test
  public void testGetExperimentTemplate() throws Exception {

    String body = loadContent("experimenttemplate/test_template_2.json");
    run(body, "application/json");

    Gson gson = new GsonBuilder().create();
    GetMethod getMethod = httpGet(TPL_PATH + "/" + TPL_NAME);
    Assert.assertEquals(Response.Status.OK.getStatusCode(),
        getMethod.getStatusCode());

    String json = getMethod.getResponseBodyAsString();
    JsonResponse jsonResponse = gson.fromJson(json, JsonResponse.class);
    Assert.assertEquals(Response.Status.OK.getStatusCode(),
        jsonResponse.getCode());

    ExperimentTemplate getExperimentTemplate =
        gson.fromJson(gson.toJson(jsonResponse.getResult()), ExperimentTemplate.class);
    Assert.assertEquals(TPL_NAME, getExperimentTemplate.getExperimentTemplateSpec().getName());
    deleteExperimentTemplate();
  }


  @Test
  public void testUpdateExperimentTemplate() throws IOException {

  }

  @Test
  public void testDeleteExperimentTemplate() throws Exception {
    String body = loadContent("experimenttemplate/test_template_2.json");
    run(body, "application/json");
    deleteExperimentTemplate();

    GetMethod getMethod = httpGet(TPL_PATH + "/" + TPL_NAME);
    Assert.assertEquals(Response.Status.NOT_FOUND.getStatusCode(),
        getMethod.getStatusCode());

  }

  @Test
  public void testListExperimentTemplates() throws IOException {

  }

  protected void deleteExperimentTemplate() throws IOException {
    Gson gson = new GsonBuilder().create();
    DeleteMethod deleteMethod = httpDelete(TPL_PATH + "/" + TPL_NAME);
    Assert.assertEquals(Response.Status.OK.getStatusCode(),
        deleteMethod.getStatusCode());

    String json = deleteMethod.getResponseBodyAsString();
    JsonResponse jsonResponse = gson.fromJson(json, JsonResponse.class);
    Assert.assertEquals(Response.Status.OK.getStatusCode(),
        jsonResponse.getCode());

    ExperimentTemplate deletedTpl =
        gson.fromJson(gson.toJson(jsonResponse.getResult()), ExperimentTemplate.class);
    Assert.assertEquals(TPL_NAME, deletedTpl.getExperimentTemplateSpec().getName());
  }

  protected void run(String body, String contentType) throws Exception {
    Gson gson = new GsonBuilder().create();

    // create
    LOG.info("Create ExperimentTemplate using ExperimentTemplate REST API");
    LOG.info(body);
    PostMethod postMethod = httpPost(TPL_PATH, body, contentType);

    LOG.info(postMethod.getResponseBodyAsString());

    Assert.assertEquals(Response.Status.OK.getStatusCode(),
        postMethod.getStatusCode());

    String json = postMethod.getResponseBodyAsString();
    JsonResponse jsonResponse = gson.fromJson(json, JsonResponse.class);
    Assert.assertEquals(Response.Status.OK.getStatusCode(),
        jsonResponse.getCode());

        ExperimentTemplate tpl =
        gson.fromJson(gson.toJson(jsonResponse.getResult()), ExperimentTemplate.class);
    verifyCreateExperimentTemplateApiResult(tpl);
  }

  protected void verifyCreateExperimentTemplateApiResult(ExperimentTemplate tpl)
      throws Exception {
    Assert.assertNotNull(tpl.getExperimentTemplateSpec().getName());
    Assert.assertNotNull(tpl.getExperimentTemplateSpec());
  }
}
