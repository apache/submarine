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

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.reflect.TypeToken;
import org.apache.submarine.commons.utils.SubmarineConfiguration;
import org.apache.submarine.server.api.experimenttemplate.ExperimentTemplate;
import org.apache.submarine.server.api.spec.ExperimentSpec;
import org.apache.submarine.server.api.spec.ExperimentTemplateParamSpec;
import org.apache.submarine.server.api.spec.ExperimentTemplateSpec;
import org.apache.submarine.server.api.spec.KernelSpec;
import org.apache.submarine.server.response.JsonResponse;
import org.junit.After;
import org.junit.Assert;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

import javax.ws.rs.core.Response;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.lang.reflect.Type;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

public class ExperimentTemplateRestApiTest {
  private static ExperimentTemplateRestApi experimentTemplateStoreApi;
  private static ExperimentTemplateSpec experimentTemplateSpec;

  /*
   * private static String experimentTemplateName = "testName"; private static
   * String experimentTemplateAuthor = "testAuthor"; private static String
   * experimentTemplateDescription = "testDescription"; private static String
   * experimentTemplateParamName = "testParamName"; private static String
   * experimentTemplateParamValue = "testParamValue"; private static String
   * experimentTemplateParamRequired = "testParamRequired"; private static String
   * experimentTemplateParamDescription = "testParamDescription";
   */

  private static GsonBuilder gsonBuilder = new GsonBuilder();
  private static Gson gson = gsonBuilder.setDateFormat("yyyy-MM-dd HH:mm:ss").create();

  @BeforeClass
  public static void init() {
    SubmarineConfiguration submarineConf = SubmarineConfiguration.getInstance();
    submarineConf.setMetastoreJdbcUrl("jdbc:mysql://127.0.0.1:3306/submarine_test?" + "useUnicode=true&amp;"
        + "characterEncoding=UTF-8&amp;" + "autoReconnect=true&amp;" + "failOverReadOnly=false&amp;"
        + "zeroDateTimeBehavior=convertToNull&amp;" + "useSSL=false");
    submarineConf.setMetastoreJdbcUserName("submarine_test");
    submarineConf.setMetastoreJdbcPassword("password_test");
    experimentTemplateStoreApi = new ExperimentTemplateRestApi();

  }

  @Before
  public void createAndUpdateExperimentTemplate() {

    String body = loadContent("experimenttemplate/test_template_1.json");
    experimentTemplateSpec = gson.fromJson(body, ExperimentTemplateSpec.class);

    // Create ExperimentTemplate
    Response createEnvResponse = experimentTemplateStoreApi.createExperimentTemplate(experimentTemplateSpec);
    assertEquals(Response.Status.OK.getStatusCode(), createEnvResponse.getStatus());

    // Update ExperimentTemplate
    experimentTemplateSpec.setDescription("newdescription");
    Response updateTplResponse = experimentTemplateStoreApi.updateExperimentTemplate(experimentTemplateSpec.getName(),
        experimentTemplateSpec);
    assertEquals(Response.Status.OK.getStatusCode(), updateTplResponse.getStatus());
  }

  @After
  public void deleteExperimentTemplate() {

    String body = loadContent("experimenttemplate/test_template_1.json");
    experimentTemplateSpec = gson.fromJson(body, ExperimentTemplateSpec.class);

    Response deleteEnvResponse = experimentTemplateStoreApi.deleteExperimentTemplate(experimentTemplateSpec.getName());
    assertEquals(Response.Status.OK.getStatusCode(), deleteEnvResponse.getStatus());
  }

  @Test
  public void getExperimentTemplate() {

    String body = loadContent("experimenttemplate/test_template_1.json");
    experimentTemplateSpec = gson.fromJson(body, ExperimentTemplateSpec.class);

    Response getEnvResponse = experimentTemplateStoreApi.getExperimentTemplate(experimentTemplateSpec.getName());
    ExperimentTemplate experimentTemplate = getExperimentTemplateFromResponse(getEnvResponse);
    assertEquals("foo", experimentTemplate.getExperimentTemplateSpec().getName());

  }

  private ExperimentTemplate getExperimentTemplateFromResponse(Response response) {
    String entity = (String) response.getEntity();
    Type type = new TypeToken<JsonResponse<ExperimentTemplate>>() {
    }.getType();
    JsonResponse<ExperimentTemplate> jsonResponse = gson.fromJson(entity, type);
    return jsonResponse.getResult();
  }

  @Test
  public void listExperimentTemplate() {
    Response getEnvResponse = experimentTemplateStoreApi.listExperimentTemplate("");
    String entity = (String) getEnvResponse.getEntity();
    JsonResponse jsonResponse = gson.fromJson(entity, JsonResponse.class);
    ExperimentTemplate[] experimentTemplates = gson.fromJson(gson.toJson(jsonResponse.getResult()),
        ExperimentTemplate[].class);
    assertEquals(1, experimentTemplates.length);

    ExperimentTemplate experimentTemplate = experimentTemplates[0];
    assertEquals("foo", experimentTemplate.getExperimentTemplateSpec().getName());
  }

  protected String loadContent(String resourceName) {
    StringBuilder content = new StringBuilder();
    InputStream inputStream = this.getClass().getClassLoader().getResourceAsStream(resourceName);
    BufferedReader r = new BufferedReader(new InputStreamReader(inputStream));
    String l;
    try {
      while ((l = r.readLine()) != null) {
        content.append(l).append("\n");
        inputStream.close();
      }
    } catch (IOException e) {
      Assert.fail(e.getMessage());
      // e.printStackTrace();
    }
    return content.toString();
  }
}
