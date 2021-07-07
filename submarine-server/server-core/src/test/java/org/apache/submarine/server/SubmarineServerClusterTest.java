/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
package org.apache.submarine.server;

import com.google.common.reflect.TypeToken;
import com.google.gson.Gson;
import com.google.gson.internal.LinkedTreeMap;
import org.apache.commons.httpclient.methods.GetMethod;
import org.apache.submarine.commons.cluster.ClusterClient;
import org.apache.submarine.commons.cluster.meta.ClusterMeta;
import org.apache.submarine.commons.cluster.meta.ClusterMetaType;
import org.apache.submarine.commons.utils.NetworkUtils;
import org.apache.submarine.commons.utils.SubmarineConfiguration;
import org.apache.submarine.server.response.JsonResponse;
import org.apache.submarine.server.rest.RestConstants;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Ignore;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.ws.rs.core.Response;
import java.io.IOException;
import java.lang.reflect.Constructor;
import java.lang.reflect.Type;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static java.lang.Thread.sleep;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

public class SubmarineServerClusterTest extends AbstractSubmarineServerTest {
  private static final Logger LOG = LoggerFactory.getLogger(SubmarineServerClusterTest.class);

  private static ClusterClient clusterClient = null;

  @BeforeClass
  public static void start() throws Exception {
    LOG.info("SubmarineServerClusterTest:start()");

    SubmarineConfiguration conf = SubmarineConfiguration.getInstance();
    String serverHost = NetworkUtils.findAvailableHostAddress();
    int serverPort = NetworkUtils.findRandomAvailablePortOnAllLocalInterfaces();
    String clusterAdd = serverHost + ":" + serverPort;
    conf.setClusterAddress(clusterAdd);

    // Run the workbench service in a thread
    startUp(SubmarineServerClusterTest.class.getSimpleName());

    // Mock Cluster client
    Class clazz = ClusterClient.class;
    Constructor constructor = null;
    constructor = clazz.getDeclaredConstructor();
    constructor.setAccessible(true);
    clusterClient = (ClusterClient) constructor.newInstance();
    clusterClient.start(SubmarineServerClusterTest.class.getSimpleName());

    // Waiting for cluster startup
    int wait = 0;
    while (wait++ < 100) {
      if (clusterClient.raftInitialized()) {
        LOG.info("SubmarineServerClusterTest::start {}(ms) found cluster leader", wait * 3000);
        break;
      }

      sleep(3000);
    }

    assertTrue("Can not start Submarine server!", clusterClient.raftInitialized());

    // Waiting for the submarine server to register in the cluster and client send heartbeat to cluster
    sleep(10000);
  }

  @AfterClass
  public static void stop() throws Exception {
    LOG.info("SubmarineServerClusterTest::stop >>>");
    shutDown();

    if (null != clusterClient) {
      clusterClient.shutdown();
    }
    LOG.info("SubmarineServerClusterTest::stop <<<");
  }

  @Ignore
  @Test
  public void testGetServerClusterMeta() {
    LOG.info("SubmarineServerClusterTest::testGetServerClusterMeta >>>");
    // Get metadata for workbench server
    Object srvMeta = clusterClient.getClusterMeta(ClusterMetaType.SERVER_META, "");
    LOG.info("testGetWorkbenchClusterMeta = {}", srvMeta.toString());

    assertNotNull(srvMeta);
    assertEquals(true, (srvMeta instanceof HashMap));
    HashMap hashMap = (HashMap) srvMeta;

    assertEquals(hashMap.size(), 1);
    LOG.info("SubmarineServerClusterTest::testGetServerClusterMeta <<<");
  }

  @Ignore
  @Test
  public void testGetClusterAddress() throws IOException {
    GetMethod response = httpGet("/api/" + RestConstants.V1 + "/"
        + RestConstants.CLUSTER + "/" + RestConstants.ADDRESS);
    LOG.info(response.toString());

    String requestBody = response.getResponseBodyAsString();
    LOG.info(requestBody);

    Type type = new TypeToken<JsonResponse<List<String>>>() {}.getType();
    Gson gson = new Gson();
    JsonResponse<List<String>> jsonResponse = gson.fromJson(requestBody, type);
    LOG.info(jsonResponse.getResult().toString());
    assertEquals(jsonResponse.getCode(), Response.Status.OK.getStatusCode());

    List<String> listAddr = jsonResponse.getResult();
    LOG.info("listAddr.size = {}", listAddr.size());
    assertEquals(listAddr.size(), 1);
  }

  private ArrayList<HashMap<String, Object>> getClusterNodes() throws IOException {
    GetMethod response = httpGet("/api/" + RestConstants.V1 + "/"
        + RestConstants.CLUSTER + "/" + RestConstants.NODES);
    LOG.info(response.toString());

    String requestBody = response.getResponseBodyAsString();
    LOG.info(requestBody);

    Type type = new TypeToken<JsonResponse<ArrayList<HashMap<String, Object>>>>() {}.getType();
    Gson gson = new Gson();
    JsonResponse<ArrayList<HashMap<String, Object>>> jsonResponse = gson.fromJson(requestBody, type);
    LOG.info(jsonResponse.getResult().toString());
    assertEquals(jsonResponse.getCode(), Response.Status.OK.getStatusCode());

    ArrayList<HashMap<String, Object>> listNodes = jsonResponse.getResult();
    LOG.info("listNodes.size = {}", listNodes.size());
    assertEquals(listNodes.size(), 1);

    return listNodes;
  }

  @Ignore
  @Test
  public void testGetClusterNodes() throws IOException {
    getClusterNodes();
  }

  @Ignore
  @Test
  public void testGetClusterNode() throws IOException {
    ArrayList<HashMap<String, Object>> listNodes = getClusterNodes();

    Map<String, Object> properties
        = (LinkedTreeMap<String, Object>) listNodes.get(0).get(ClusterMeta.PROPERTIES);
    ArrayList<String> intpList = (ArrayList<String>) properties.get(ClusterMeta.INTP_PROCESS_LIST);
    String nodeName = listNodes.get(0).get(ClusterMeta.NODE_NAME).toString();
    String intpName = intpList.get(0);
    LOG.info("properties = {}", properties);
    LOG.info("intpList = {}", intpList);
    LOG.info("nodeName = {}", nodeName);
    LOG.info("intpName = {}", intpName);

    GetMethod response = httpGet("/api/" + RestConstants.V1 + "/"
        + RestConstants.CLUSTER + "/" + RestConstants.NODE + "/" + nodeName + "/" + intpName);
    LOG.info(response.toString());

    String requestBody = response.getResponseBodyAsString();
    LOG.info(requestBody);

    Type type = new TypeToken<JsonResponse<ArrayList<HashMap<String, Object>>>>() {}.getType();
    Gson gson = new Gson();
    JsonResponse<ArrayList<HashMap<String, Object>>> jsonResponse = gson.fromJson(requestBody, type);
    LOG.info(jsonResponse.getResult().toString());
    assertEquals(jsonResponse.getCode(), Response.Status.OK.getStatusCode());

    ArrayList<HashMap<String, Object>> intpProcesses = jsonResponse.getResult();
    LOG.info("intpProcesses = {}", intpProcesses);
    assertEquals(intpProcesses.size(), 1);
  }
}
