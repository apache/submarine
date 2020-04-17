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

import javax.ws.rs.core.MediaType;
import javax.ws.rs.core.Response;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.Date;
import java.util.HashMap;
import java.util.Map;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonObject;
import io.kubernetes.client.ApiClient;
import io.kubernetes.client.ApiException;
import io.kubernetes.client.Configuration;
import io.kubernetes.client.JSON;
import io.kubernetes.client.apis.CustomObjectsApi;
import io.kubernetes.client.util.ClientBuilder;
import io.kubernetes.client.util.KubeConfig;
import org.apache.commons.httpclient.methods.DeleteMethod;
import org.apache.commons.httpclient.methods.GetMethod;
import org.apache.commons.httpclient.methods.PostMethod;
import org.apache.commons.io.FileUtils;
import org.apache.submarine.server.AbstractSubmarineServerTest;
import org.apache.submarine.server.api.job.Job;
import org.apache.submarine.server.api.job.JobId;
import org.apache.submarine.server.json.JobIdDeserializer;
import org.apache.submarine.server.json.JobIdSerializer;
import org.apache.submarine.server.response.JsonResponse;
import org.apache.submarine.server.rest.RestConstants;
import org.joda.time.DateTime;
import org.junit.Assert;
import org.junit.BeforeClass;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

@SuppressWarnings("rawtypes")
public class JobManagerRestApiIT extends AbstractSubmarineServerTest {
  private static final Logger LOG = LoggerFactory.getLogger(JobManagerRestApiIT.class);

  private static CustomObjectsApi k8sApi;
  /** Key is the ml framework name, the value is the operator */
  private static Map<String, KfOperator> kfOperatorMap;
  private static String JOB_PATH = "/api/" + RestConstants.V1 + "/" + RestConstants.JOBS;

  private Gson gson = new GsonBuilder()
      .registerTypeAdapter(JobId.class, new JobIdSerializer())
      .registerTypeAdapter(JobId.class, new JobIdDeserializer())
      .create();

  @BeforeClass
  public static void startUp() throws IOException {
    Assert.assertTrue(checkIfServerIsRunning());

    // The kube config path defined by kind-cluster-build.sh
    String confPath = System.getProperty("user.home") + "/.kube/kind-config-kind";
    KubeConfig config = KubeConfig.loadKubeConfig(new FileReader(confPath));
    ApiClient client = ClientBuilder.kubeconfig(config).build();
    Configuration.setDefaultApiClient(client);
    k8sApi = new CustomObjectsApi();

    kfOperatorMap = new HashMap<>();
    kfOperatorMap.put("tensorflow", new KfOperator("v1", "tfjobs"));
    kfOperatorMap.put("pytorch", new KfOperator("v1", "pytorchjobs"));
  }

  @Test
  public void testJobServerPing() throws IOException {
    GetMethod response = httpGet("/api/" + RestConstants.V1 + "/"
        + RestConstants.JOBS + "/" + RestConstants.PING);
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

  private void run(String body, String patchBody, String contentType) throws Exception {
    // create
    LOG.info("Create training job by Job REST API");
    PostMethod postMethod = httpPost(JOB_PATH, body, contentType);
    Assert.assertEquals(Response.Status.OK.getStatusCode(), postMethod.getStatusCode());

    String json = postMethod.getResponseBodyAsString();
    JsonResponse jsonResponse = gson.fromJson(json, JsonResponse.class);
    Assert.assertEquals(Response.Status.OK.getStatusCode(), jsonResponse.getCode());

    Job createdJob = gson.fromJson(gson.toJson(jsonResponse.getResult()), Job.class);
    verifyCreateJobApiResult(createdJob);

    // find
    GetMethod getMethod = httpGet(JOB_PATH + "/" + createdJob.getJobId().toString());
    Assert.assertEquals(Response.Status.OK.getStatusCode(), getMethod.getStatusCode());

    json = getMethod.getResponseBodyAsString();
    jsonResponse = gson.fromJson(json, JsonResponse.class);
    Assert.assertEquals(Response.Status.OK.getStatusCode(), jsonResponse.getCode());

    Job foundJob = gson.fromJson(gson.toJson(jsonResponse.getResult()), Job.class);
    verifyGetJobApiResult(createdJob, foundJob);

    // patch
    // TODO(jiwq): the commons-httpclient not support patch method
    // https://tools.ietf.org/html/rfc5789

    // delete
    DeleteMethod deleteMethod = httpDelete(JOB_PATH + "/" + createdJob.getJobId().toString());
    Assert.assertEquals(Response.Status.OK.getStatusCode(), deleteMethod.getStatusCode());

    json = deleteMethod.getResponseBodyAsString();
    jsonResponse = gson.fromJson(json, JsonResponse.class);
    Assert.assertEquals(Response.Status.OK.getStatusCode(), jsonResponse.getCode());

    Job deletedJob = gson.fromJson(gson.toJson(jsonResponse.getResult()), Job.class);
    verifyDeleteJobApiResult(createdJob, deletedJob);
  }

  private void verifyCreateJobApiResult(Job createdJob) throws Exception {
    Assert.assertNotNull(createdJob.getUid());
    Assert.assertNotNull(createdJob.getAcceptedTime());
    Assert.assertEquals(Job.Status.STATUS_ACCEPTED.getValue(), createdJob.getStatus());

    assertK8sResultEquals(createdJob);
  }

  private void verifyGetJobApiResult(Job createdJob, Job foundJob) throws Exception {
    Assert.assertEquals(createdJob.getJobId(), foundJob.getJobId());
    Assert.assertEquals(createdJob.getUid(), foundJob.getUid());
    Assert.assertEquals(createdJob.getName(), foundJob.getName());
    Assert.assertEquals(createdJob.getAcceptedTime(), foundJob.getAcceptedTime());

    assertK8sResultEquals(foundJob);
  }

  private void assertK8sResultEquals(Job job) throws Exception {
    KfOperator operator = kfOperatorMap.get(job.getSpec().getLibrarySpec().getName()
        .toLowerCase());
    JsonObject rootObject = getJobByK8sApi(operator.getGroup(), operator.getVersion(),
        operator.getNamespace(), operator.getPlural(), job.getName());
    JsonObject metadataObject = rootObject.getAsJsonObject("metadata");

    String uid = metadataObject.getAsJsonPrimitive("uid").getAsString();
    LOG.info("Uid from Job REST is {}", job.getUid());
    LOG.info("Uid from K8s REST is {}", uid);
    Assert.assertEquals(job.getUid(), uid);

    String creationTimestamp = metadataObject.getAsJsonPrimitive("creationTimestamp")
        .getAsString();
    Date expectedDate = new DateTime(job.getAcceptedTime()).toDate();
    Date actualDate = new DateTime(creationTimestamp).toDate();
    LOG.info("CreationTimestamp from Job REST is {}", expectedDate);
    LOG.info("CreationTimestamp from K8s REST is {}", actualDate);
    Assert.assertEquals(expectedDate, actualDate);
  }

  private void verifyDeleteJobApiResult(Job createdJob, Job deletedJob) {
    Assert.assertEquals(createdJob.getName(), deletedJob.getName());
    Assert.assertEquals(Job.Status.STATUS_DELETED.getValue(), deletedJob.getStatus());

    // verify the result by K8s api
    KfOperator operator = kfOperatorMap.get(createdJob.getSpec().getLibrarySpec().getName()
        .toLowerCase());
    JsonObject rootObject = null;
    try {
      rootObject = getJobByK8sApi(operator.getGroup(), operator.getVersion(),
          operator.getNamespace(), operator.getPlural(), createdJob.getName());
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
    PostMethod postMethod = httpPost(JOB_PATH, "", MediaType.APPLICATION_JSON);
    Assert.assertEquals(Response.Status.OK.getStatusCode(), postMethod.getStatusCode());

    String json = postMethod.getResponseBodyAsString();
    JsonResponse jsonResponse = gson.fromJson(json, JsonResponse.class);
    Assert.assertEquals(Response.Status.OK.getStatusCode(), jsonResponse.getCode());
  }

  @Test
  public void testNotFoundJob() throws Exception {
    GetMethod getMethod = httpGet(JOB_PATH + "/" + "job_123456789_0001");
    Assert.assertEquals(Response.Status.NOT_FOUND.getStatusCode(), getMethod.getStatusCode());

    String json = getMethod.getResponseBodyAsString();
    JsonResponse jsonResponse = gson.fromJson(json, JsonResponse.class);
    Assert.assertEquals(Response.Status.NOT_FOUND.getStatusCode(), jsonResponse.getCode());
  }

  String loadContent(String resourceName) throws Exception {
    URL fileUrl = this.getClass().getResource("/" + resourceName);
    LOG.info("Resource file: " + fileUrl);
    return FileUtils.readFileToString(new File(fileUrl.toURI()), StandardCharsets.UTF_8);
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
