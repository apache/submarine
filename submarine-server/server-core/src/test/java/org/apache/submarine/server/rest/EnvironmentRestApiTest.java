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
import org.apache.submarine.server.api.environment.Environment;
import org.apache.submarine.server.api.environment.EnvironmentId;
import org.apache.submarine.server.api.spec.EnvironmentSpec;
import org.apache.submarine.server.api.spec.KernelSpec;
import org.apache.submarine.server.gson.EnvironmentIdDeserializer;
import org.apache.submarine.server.gson.EnvironmentIdSerializer;
import org.apache.submarine.server.response.JsonResponse;
import org.junit.After;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

import javax.ws.rs.core.Response;
import java.lang.reflect.Type;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;

public class EnvironmentRestApiTest {
  private static EnvironmentRestApi environmentStoreApi;
  private static String kernelName = "team_default_python_3";
  private static String dockerImage = "continuumio/anaconda3";
  private static List<String> kernelChannels = Arrays.asList("defaults", "anaconda");
  private static List<String> kernelCondaDependencies = Arrays.asList(
      "_ipyw_jlab_nb_ext_conf=0.1.0=py37_0",
      "alabaster=0.7.12=py37_0",
      "anaconda=2020.02=py37_0",
      "anaconda-client=1.7.2=py37_0",
      "anaconda-navigator=1.9.12=py37_0");
  private static List<String> kernelPipDependencies = Arrays.asList(
      "apache-submarine==0.5.0",
      "pyarrow==0.17.0");

  private static GsonBuilder gsonBuilder = new GsonBuilder()
      .registerTypeAdapter(EnvironmentId.class, new EnvironmentIdSerializer())
      .registerTypeAdapter(EnvironmentId.class, new EnvironmentIdDeserializer());
  private static Gson gson = gsonBuilder.setDateFormat("yyyy-MM-dd HH:mm:ss").create();

  @BeforeClass
  public static void init() {
    SubmarineConfiguration submarineConf = SubmarineConfiguration.getInstance();
    submarineConf.setMetastoreJdbcUrl("jdbc:mysql://127.0.0.1:3306/submarine_test?" +
        "useUnicode=true&amp;" +
        "characterEncoding=UTF-8&amp;" +
        "autoReconnect=true&amp;" +
        "failOverReadOnly=false&amp;" +
        "zeroDateTimeBehavior=convertToNull&amp;" +
        "useSSL=false");
    submarineConf.setMetastoreJdbcUserName("submarine_test");
    submarineConf.setMetastoreJdbcPassword("password_test");
    environmentStoreApi = new EnvironmentRestApi();
  }

  @Before
  public void createAndUpdateEnvironment() {
    KernelSpec kernelSpec = new KernelSpec();
    kernelSpec.setName(kernelName);
    kernelSpec.setChannels(kernelChannels);
    kernelSpec.setCondaDependencies(kernelCondaDependencies);
    EnvironmentSpec environmentSpec = new EnvironmentSpec();
    environmentSpec.setDockerImage(dockerImage);
    environmentSpec.setKernelSpec(kernelSpec);
    environmentSpec.setName("foo");

    // Create Environment
    Response createEnvResponse =
        environmentStoreApi.createEnvironment(environmentSpec);
    assertEquals(Response.Status.OK.getStatusCode(),
        createEnvResponse.getStatus());

    // Update Environment
    environmentSpec.setDockerImage("continuumio/miniconda");
    Response updateEnvResponse =
        environmentStoreApi.updateEnvironment("foo", environmentSpec);
    assertEquals(Response.Status.OK.getStatusCode(),
        updateEnvResponse.getStatus());
  }

  @After
  public void deleteEnvironment() {
    Response deleteEnvResponse = environmentStoreApi
            .deleteEnvironment("foo");
    assertEquals(Response.Status.OK.getStatusCode(), deleteEnvResponse.getStatus());
  }

  @Test
  public void getEnvironment() {
    Response getEnvResponse = environmentStoreApi.getEnvironment("foo");
    Environment environment = getEnvironmentFromResponse(getEnvResponse);
    assertEquals("foo", environment.getEnvironmentSpec().getName());
    assertEquals(kernelName, environment.getEnvironmentSpec().getKernelSpec().getName());
    assertEquals(kernelChannels, environment.getEnvironmentSpec().getKernelSpec().getChannels());
    assertEquals(kernelCondaDependencies,
        environment.getEnvironmentSpec().getKernelSpec().getCondaDependencies());
    assertEquals("continuumio/miniconda", environment.getEnvironmentSpec().getDockerImage());
  }

  private Environment getEnvironmentFromResponse(Response response) {
    String entity = (String) response.getEntity();
    Type type = new TypeToken<JsonResponse<Environment>>() {}.getType();
    JsonResponse<Environment> jsonResponse = gson.fromJson(entity, type);
    return jsonResponse.getResult();
  }

  @Test
  public void listEnvironment() {
    Response getEnvResponse = environmentStoreApi.listEnvironment("");
    String entity = (String) getEnvResponse.getEntity();
    JsonResponse jsonResponse = gson.fromJson(entity, JsonResponse.class);

    // environments.length = 2; One is created in this test, one is get from database
    Environment[] environments = gson
        .fromJson(gson.toJson(jsonResponse.getResult()), Environment[].class);
    assertEquals(2, environments.length);

    Environment environment = environments[0];
    assertEquals("foo", environment.getEnvironmentSpec().getName());
    assertEquals(kernelName,
        environment.getEnvironmentSpec().getKernelSpec().getName());
    assertEquals(kernelChannels,
        environment.getEnvironmentSpec().getKernelSpec().getChannels());
    assertEquals(kernelCondaDependencies,
        environment.getEnvironmentSpec().getKernelSpec().getCondaDependencies());
    assertEquals("continuumio/miniconda",
        environment.getEnvironmentSpec().getDockerImage());
  }
}
