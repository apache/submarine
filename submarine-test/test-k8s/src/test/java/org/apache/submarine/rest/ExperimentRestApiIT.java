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

import java.io.FileReader;
import java.io.IOException;
import java.util.Date;
import java.util.HashMap;
import java.util.Map;

import javax.ws.rs.core.MediaType;
import javax.ws.rs.core.Response;

import org.apache.commons.httpclient.methods.DeleteMethod;
import org.apache.commons.httpclient.methods.GetMethod;
import org.apache.commons.httpclient.methods.PostMethod;
import org.apache.submarine.commons.utils.SubmarineConfVars;
import org.apache.submarine.commons.utils.SubmarineConfiguration;
import org.apache.submarine.server.AbstractSubmarineServerTest;
import org.apache.submarine.server.api.environment.EnvironmentId;
import org.apache.submarine.server.api.experiment.Experiment;
import org.apache.submarine.server.api.experiment.ExperimentId;
import org.apache.submarine.server.gson.EnvironmentIdDeserializer;
import org.apache.submarine.server.gson.EnvironmentIdSerializer;
import org.apache.submarine.server.gson.ExperimentIdDeserializer;
import org.apache.submarine.server.gson.ExperimentIdSerializer;
import org.apache.submarine.server.api.environment.Environment;
import org.apache.submarine.server.response.JsonResponse;
import org.apache.submarine.server.rest.RestConstants;
import org.joda.time.DateTime;
import org.junit.Assert;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;

import io.kubernetes.client.openapi.ApiClient;
import io.kubernetes.client.openapi.ApiException;
import io.kubernetes.client.openapi.Configuration;
import io.kubernetes.client.openapi.JSON;
import io.kubernetes.client.openapi.apis.CustomObjectsApi;
import io.kubernetes.client.util.ClientBuilder;
import io.kubernetes.client.util.KubeConfig;

@SuppressWarnings("rawtypes")
public class ExperimentRestApiIT extends AbstractSubmarineServerTest {
  private static final Logger LOG = LoggerFactory.getLogger(ExperimentRestApiIT.class);

  private static CustomObjectsApi k8sApi;
  /**
   * Key is the ml framework name, the value is the operator
   */
  private static Map<String, KfOperator> kfOperatorMap;
  private static final String BASE_API_PATH = "/api/" + RestConstants.V1 + "/" + RestConstants.EXPERIMENT;
  private static final String LOG_API_PATH = BASE_API_PATH + "/" + RestConstants.LOGS;

  private final Gson gson = new GsonBuilder()
      .registerTypeAdapter(ExperimentId.class, new ExperimentIdSerializer())
      .registerTypeAdapter(ExperimentId.class, new ExperimentIdDeserializer())
      .registerTypeAdapter(EnvironmentId.class, new EnvironmentIdSerializer())
      .registerTypeAdapter(EnvironmentId.class, new EnvironmentIdDeserializer())
      .create();

  private static SubmarineConfiguration conf =
      SubmarineConfiguration.getInstance();

  @BeforeClass
  public static void startUp() throws IOException {
    Assert.assertTrue(checkIfServerIsRunning());

    // The kube config is created when the cluster builds
    String confPath = System.getProperty("user.home") + "/.kube/config";
    KubeConfig config = KubeConfig.loadKubeConfig(new FileReader(confPath));
    ApiClient client = ClientBuilder.kubeconfig(config).build();
    Configuration.setDefaultApiClient(client);
    k8sApi = new CustomObjectsApi();

    kfOperatorMap = new HashMap<>();
    kfOperatorMap.put("tensorflow", new KfOperator("v1", "tfjobs"));
    kfOperatorMap.put("pytorch", new KfOperator("v1", "pytorchjobs"));
  }

  @Before
  public void setUp() throws Exception {
    Thread.sleep(5000); // timeout for each case, ensuring k8s-client has enough time to delete resources
  }

  @Test
  public void testJobServerPing() throws IOException {
    GetMethod response = httpGet("/api/" + RestConstants.V1 + "/"
        + RestConstants.EXPERIMENT + "/" + RestConstants.PING);
    String requestBody = response.getResponseBodyAsString();
    Gson gson = new Gson();
    JsonResponse jsonResponse = gson.fromJson(requestBody, JsonResponse.class);
    Assert.assertEquals(Response.Status.OK.getStatusCode(), jsonResponse.getCode());
    Assert.assertEquals("Pong", jsonResponse.getResult().toString());
  }

  @Test
  public void testTensorFlowWithJsonSpec() throws Exception {
    String body = loadContent("tensorflow/tf-mnist-req.json");
    String patchBody = loadContent("tensorflow/tf-mnist-patch-req.json");
    run(body, patchBody, "application/json");
  }

  @Test
  public void testTensorFlowWithYamlSpec() throws Exception {
    String body = loadContent("tensorflow/tf-mnist-req.yaml");
    String patchBody = loadContent("tensorflow/tf-mnist-patch-req.yaml");
    run(body, patchBody, "application/yaml");
  }

  @Test
  public void testTensorFlowUsingEnvWithJsonSpec() throws Exception {
    Gson gson = new GsonBuilder()
        .registerTypeAdapter(EnvironmentId.class, new EnvironmentIdSerializer())
        .registerTypeAdapter(EnvironmentId.class, new EnvironmentIdDeserializer())
        .create();
    // Create environment
    String envBody = loadContent("environment/test_env_1.json");
    run(envBody, "application/json");

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

    String body = loadContent("tensorflow/tf-mnist-with-env-req.json");
    String patchBody =
        loadContent("tensorflow/tf-mnist-with-env-patch-req.json");
    run(getEnvironment, body, patchBody, "application/json");

    // Delete environment
    deleteEnvironment();
  }

  @Test
  public void testPyTorchWithJsonSpec() throws Exception {
    String body = loadContent("pytorch/pt-mnist-req.json");
    String patchBody = loadContent("pytorch/pt-mnist-patch-req.json");
    run(body, patchBody, "application/json");
  }

  @Test
  public void testPyTorchWithYamlSpec() throws Exception {
    String body = loadContent("pytorch/pt-mnist-req.yaml");
    String patchBody = loadContent("pytorch/pt-mnist-patch-req.yaml");
    run(body, patchBody, "application/yaml");
  }

  @Test
  public void testTensorFlowUsingCodeWithJsonSpec() throws Exception {
    String body = loadContent("tensorflow/tf-mnist-with-http-git-code-localizer-req.json");
    String patchBody = loadContent("tensorflow/tf-mnist-with-http-git-code-localizer-req.json");
    run(body, patchBody, "application/json");
  }

  @Test
  public void testTensorFlowUsingSSHCodeWithJsonSpec() throws Exception {
    String body = loadContent("tensorflow/tf-mnist-with-ssh-git-code-localizer-req.json");
    String patchBody = loadContent("tensorflow/tf-mnist-with-ssh-git-code-localizer-req.json");
    run(body, patchBody, "application/json");
  }

  private void run(String body, String patchBody, String contentType) throws Exception {
    // create
    LOG.info("Create training job by Job REST API");
    PostMethod postMethod = httpPost(BASE_API_PATH, body, contentType);
    Assert.assertEquals(Response.Status.OK.getStatusCode(), postMethod.getStatusCode());

    String json = postMethod.getResponseBodyAsString();
    JsonResponse jsonResponse = gson.fromJson(json, JsonResponse.class);
    Assert.assertEquals(Response.Status.OK.getStatusCode(), jsonResponse.getCode());
    Experiment createdExperiment = gson.fromJson(gson.toJson(jsonResponse.getResult()), Experiment.class);
    verifyCreateJobApiResult(createdExperiment);

    // find
    GetMethod getMethod = httpGet(BASE_API_PATH + "/" +
            createdExperiment.getSpec().getMeta().getExperimentId());
    Assert.assertEquals(Response.Status.OK.getStatusCode(), getMethod.getStatusCode());

    json = getMethod.getResponseBodyAsString();
    jsonResponse = gson.fromJson(json, JsonResponse.class);
    Assert.assertEquals(Response.Status.OK.getStatusCode(), jsonResponse.getCode());

    Experiment foundExperiment = gson.fromJson(gson.toJson(jsonResponse.getResult()), Experiment.class);
    verifyGetJobApiResult(createdExperiment, foundExperiment);

    // get log list
    // TODO(JohnTing): Test the job log after creating the job

    // patch
    // TODO(jiwq): the commons-httpclient not support patch method
    // https://tools.ietf.org/html/rfc5789

    // delete
    DeleteMethod deleteMethod = httpDelete(BASE_API_PATH + "/" +
            createdExperiment.getSpec().getMeta().getExperimentId());
    Assert.assertEquals(Response.Status.OK.getStatusCode(), deleteMethod.getStatusCode());

    json = deleteMethod.getResponseBodyAsString();
    jsonResponse = gson.fromJson(json, JsonResponse.class);
    Assert.assertEquals(Response.Status.OK.getStatusCode(), jsonResponse.getCode());

    Experiment deletedExperiment = gson.fromJson(gson.toJson(jsonResponse.getResult()), Experiment.class);
    verifyDeleteJobApiResult(createdExperiment, deletedExperiment);
  }

  private void run(Environment expectedEnv, String body, String patchBody,
                   String contentType) throws Exception {
    // create
    LOG.info("Create training job using Environment by Job REST API");
    PostMethod postMethod = httpPost(BASE_API_PATH, body, contentType);
    Assert.assertEquals(Response.Status.OK.getStatusCode(),
        postMethod.getStatusCode());

    String json = postMethod.getResponseBodyAsString();
    JsonResponse jsonResponse = gson.fromJson(json, JsonResponse.class);
    Assert.assertEquals(Response.Status.OK.getStatusCode(),
        jsonResponse.getCode());

    Experiment createdExperiment =
        gson.fromJson(gson.toJson(jsonResponse.getResult()), Experiment.class);
    verifyCreateJobApiResult(expectedEnv, createdExperiment);

    // find
    GetMethod getMethod =
        httpGet(BASE_API_PATH + "/" + createdExperiment.getSpec().getMeta().getExperimentId());
    Assert.assertEquals(Response.Status.OK.getStatusCode(),
        getMethod.getStatusCode());

    json = getMethod.getResponseBodyAsString();
    jsonResponse = gson.fromJson(json, JsonResponse.class);
    Assert.assertEquals(Response.Status.OK.getStatusCode(),
        jsonResponse.getCode());

    Experiment foundExperiment =
        gson.fromJson(gson.toJson(jsonResponse.getResult()), Experiment.class);
    verifyGetJobApiResult(createdExperiment, foundExperiment);

    // delete
    DeleteMethod deleteMethod =
        httpDelete(BASE_API_PATH + "/" + createdExperiment.getSpec().getMeta().getExperimentId());
    Assert.assertEquals(Response.Status.OK.getStatusCode(),
        deleteMethod.getStatusCode());

    json = deleteMethod.getResponseBodyAsString();
    jsonResponse = gson.fromJson(json, JsonResponse.class);
    Assert.assertEquals(Response.Status.OK.getStatusCode(),
        jsonResponse.getCode());

    Experiment deletedExperiment =
        gson.fromJson(gson.toJson(jsonResponse.getResult()), Experiment.class);
    verifyDeleteJobApiResult(createdExperiment, deletedExperiment);
  }

  private void verifyCreateJobApiResult(Experiment createdExperiment) throws Exception {
    Assert.assertNotNull(createdExperiment.getUid());
    Assert.assertNotNull(createdExperiment.getAcceptedTime());
    Assert.assertEquals(Experiment.Status.STATUS_ACCEPTED.getValue(), createdExperiment.getStatus());

    assertK8sResultEquals(createdExperiment);
  }

  private void verifyCreateJobApiResult(Environment env, Experiment createdJob)
      throws Exception {
    Assert.assertNotNull(createdJob.getUid());
    Assert.assertNotNull(createdJob.getAcceptedTime());
    Assert.assertEquals(Experiment.Status.STATUS_ACCEPTED.getValue(),
        createdJob.getStatus());

    assertK8sResultEquals(env, createdJob);
  }

  private void verifyGetJobApiResult(
      Experiment createdExperiment, Experiment foundExperiment) throws Exception {
    Assert.assertEquals(createdExperiment.getExperimentId(), foundExperiment.getExperimentId());
    Assert.assertEquals(createdExperiment.getUid(), foundExperiment.getUid());
    Assert.assertEquals(createdExperiment.getSpec().getMeta().getExperimentId(),
            foundExperiment.getSpec().getMeta().getExperimentId());
    Assert.assertEquals(createdExperiment.getAcceptedTime(), foundExperiment.getAcceptedTime());

    assertK8sResultEquals(foundExperiment);
  }

  private void assertK8sResultEquals(Environment env, Experiment experiment) throws Exception {
    KfOperator operator = kfOperatorMap.get(experiment.getSpec().getMeta().getFramework().toLowerCase());
    JsonObject rootObject =
        getJobByK8sApi(operator.getGroup(), operator.getVersion(),
            operator.getNamespace(), operator.getPlural(), experiment.getSpec().getMeta().getExperimentId());
    JsonArray actualCommand = (JsonArray) rootObject.getAsJsonObject("spec")
        .getAsJsonObject("tfReplicaSpecs").getAsJsonObject("Worker")
        .getAsJsonObject("template").getAsJsonObject("spec")
        .getAsJsonArray("initContainers").get(0).getAsJsonObject()
        .get("command");

    JsonArray expected = new JsonArray();
    expected.add("/bin/bash");
    expected.add("-c");

    String minVersion = "minVersion=\""
        + conf.getString(
        SubmarineConfVars.ConfVars.ENVIRONMENT_CONDA_MIN_VERSION)
        + "\";";
    String maxVersion = "maxVersion=\""
        + conf.getString(
        SubmarineConfVars.ConfVars.ENVIRONMENT_CONDA_MAX_VERSION)
        + "\";";
    String currentVersion = "currentVersion=$(conda -V | cut -f2 -d' ');";
    String versionCommand =
        minVersion + maxVersion + currentVersion
            + "if [ \"$(printf '%s\\n' \"$minVersion\" \"$maxVersion\" "
            + "\"$currentVersion\" | sort -V | head -n2 | tail -1 )\" "
            + "!= \"$currentVersion\" ]; then echo \"Conda version " +
            "should be between minVersion=\"4.0.1\"; " +
            "and maxVersion=\"4.11.10\";\"; exit 1; else echo "
            + "\"Conda current version is " + currentVersion + ". "
            + "Moving forward with env creation and activation.\"; "
            + "fi && ";

    String initialCommand =
        "conda create -n " + env.getEnvironmentSpec().getKernelSpec().getName();

    String channels = "";
    for (String channel : env.getEnvironmentSpec().getKernelSpec()
        .getChannels()) {
      channels += " -c " + channel;
    }

    String dependencies = "";
    for (String dependency : env.getEnvironmentSpec().getKernelSpec()
        .getCondaDependencies()) {
      dependencies += " " + dependency;
    }

    String fullCommand = versionCommand + initialCommand + channels
        + dependencies + " && echo \"source activate "
        + env.getEnvironmentSpec().getKernelSpec().getName() + "\" > ~/.bashrc"
        + " && PATH=/opt/conda/envs/env/bin:$PATH";
    expected.add(fullCommand);
    Assert.assertEquals(expected, actualCommand);

    JsonObject metadataObject = rootObject.getAsJsonObject("metadata");

    String uid = metadataObject.getAsJsonPrimitive("uid").getAsString();
    LOG.info("Uid from Job REST is {}", experiment.getUid());
    LOG.info("Uid from K8s REST is {}", uid);
    Assert.assertEquals(experiment.getUid(), uid);

    String creationTimestamp =
        metadataObject.getAsJsonPrimitive("creationTimestamp").getAsString();
    Date expectedDate = new DateTime(experiment.getAcceptedTime()).toDate();
    Date actualDate = new DateTime(creationTimestamp).toDate();
    LOG.info("CreationTimestamp from Job REST is {}", expectedDate);
    LOG.info("CreationTimestamp from K8s REST is {}", actualDate);
    Assert.assertEquals(expectedDate, actualDate);
  }

  private void assertK8sResultEquals(Experiment experiment) throws Exception {
    KfOperator operator = kfOperatorMap.get(experiment.getSpec().getMeta().getFramework().toLowerCase());
    JsonObject rootObject = getJobByK8sApi(operator.getGroup(), operator.getVersion(),
        operator.getNamespace(), operator.getPlural(), experiment.getSpec().getMeta().getExperimentId());
    JsonObject metadataObject = rootObject.getAsJsonObject("metadata");

    String uid = metadataObject.getAsJsonPrimitive("uid").getAsString();
    LOG.info("Uid from Experiment REST is {}", experiment.getUid());
    LOG.info("Uid from K8s REST is {}", uid);
    Assert.assertEquals(experiment.getUid(), uid);

    String creationTimestamp = metadataObject.getAsJsonPrimitive("creationTimestamp")
        .getAsString();
    Date expectedDate = new DateTime(experiment.getAcceptedTime()).toDate();
    Date actualDate = new DateTime(creationTimestamp).toDate();
    LOG.info("CreationTimestamp from Experiment REST is {}", expectedDate);
    LOG.info("CreationTimestamp from K8s REST is {}", actualDate);
    Assert.assertEquals(expectedDate, actualDate);
  }

  private void verifyDeleteJobApiResult(Experiment createdExperiment, Experiment deletedExperiment) {
    Assert.assertEquals(createdExperiment.getSpec().getMeta().getExperimentId(),
            deletedExperiment.getSpec().getMeta().getExperimentId());
    Assert.assertEquals(Experiment.Status.STATUS_DELETED.getValue(), deletedExperiment.getStatus());

    // verify the result by K8s api
    KfOperator operator = kfOperatorMap.get(createdExperiment.getSpec().getMeta().getFramework()
        .toLowerCase());
    JsonObject rootObject = null;
    try {
      rootObject = getJobByK8sApi(operator.getGroup(), operator.getVersion(),
          operator.getNamespace(), operator.getPlural(),
              createdExperiment.getSpec().getMeta().getExperimentId());
    } catch (ApiException e) {
      Assert.assertEquals(Response.Status.NOT_FOUND.getStatusCode(), e.getCode());
    } finally {
      Assert.assertNull(rootObject);
    }
  }

  private JsonObject getJobByK8sApi(String group, String version, String namespace, String plural,
                                    String name) throws ApiException {
    Object obj = k8sApi.getNamespacedCustomObject(group, version, namespace, plural, name);
    Gson gson = new JSON().getGson();
    JsonObject rootObject = gson.toJsonTree(obj).getAsJsonObject();
    Assert.assertNotNull("Parse the K8s API Server response failed.", rootObject);
    return rootObject;
  }

  @Test
  public void testCreateJobWithInvalidSpec() throws Exception {
    PostMethod postMethod = httpPost(BASE_API_PATH, "", MediaType.APPLICATION_JSON);
    Assert.assertEquals(Response.Status.OK.getStatusCode(), postMethod.getStatusCode());

    String json = postMethod.getResponseBodyAsString();
    JsonResponse jsonResponse = gson.fromJson(json, JsonResponse.class);
    Assert.assertEquals(Response.Status.OK.getStatusCode(), jsonResponse.getCode());
  }

  @Test
  public void testNotFoundJob() throws Exception {
    GetMethod getMethod = httpGet(BASE_API_PATH + "/" + "experiment_123456789_0001");
    Assert.assertEquals(Response.Status.NOT_FOUND.getStatusCode(), getMethod.getStatusCode());

    String json = getMethod.getResponseBodyAsString();
    JsonResponse jsonResponse = gson.fromJson(json, JsonResponse.class);
    Assert.assertEquals(Response.Status.NOT_FOUND.getStatusCode(), jsonResponse.getCode());
  }

  @Test
  public void testListJobLog() throws Exception {
    GetMethod getMethod = httpGet(LOG_API_PATH);
    Assert.assertEquals(Response.Status.OK.getStatusCode(), getMethod.getStatusCode());

    String json = getMethod.getResponseBodyAsString();
    JsonResponse jsonResponse = gson.fromJson(json, JsonResponse.class);
    Assert.assertEquals(Response.Status.OK.getStatusCode(), jsonResponse.getCode());
  }

  /**
   * Direct used by K8s api. It storage the operator's base info.
   */
  private static class KfOperator {
    private String version;
    private String plural;

    KfOperator(String version, String plural) {
      this.version = version;
      this.plural = plural;
    }

    public String getGroup() {
      return "kubeflow.org";
    }

    public String getNamespace() {
      return "default";
    }

    public String getVersion() {
      return version;
    }

    public String getPlural() {
      return plural;
    }
  }
}
