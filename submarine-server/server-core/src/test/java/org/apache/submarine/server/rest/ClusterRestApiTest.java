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
import com.google.gson.JsonElement;
import com.google.gson.JsonArray;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.google.gson.internal.LinkedTreeMap;
import org.apache.submarine.commons.cluster.meta.ClusterMeta;
import org.apache.submarine.commons.cluster.meta.ClusterMetaType;
import org.apache.submarine.commons.utils.SubmarineConfiguration;
import org.apache.submarine.commons.cluster.ClusterServer;
import org.junit.Test;
import org.junit.BeforeClass;

import javax.ws.rs.core.Response;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.apache.submarine.commons.cluster.meta.ClusterMetaType.SERVER_META;
import static org.junit.Assert.assertEquals;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class ClusterRestApiTest {
  private static ClusterServer mockClusterServer;
  private static ClusterRestApi clusterRestApi;

  private static final GsonBuilder gsonBuilder = new GsonBuilder();

  private static Gson gson = gsonBuilder.setDateFormat("yyyy-MM-dd HH:mm:ss").create();

  private static String dummyHost = "1.3.5.7";
  private static int dummyPort = 2468;

  @BeforeClass
  public static void init() {
    mockClusterServer = mock(ClusterServer.class);
    clusterRestApi = new ClusterRestApi();
    clusterRestApi.setClusterServer(mockClusterServer);
  }

  @Test
  public void testGetClusterAddress() {
    SubmarineConfiguration conf = SubmarineConfiguration.getInstance();
    String addr1 = "0.1.2.3:4569";
    String addr2 = "4.5.6.7:8888";
    conf.setClusterAddress(addr1 + "," + addr2);

    Response response = clusterRestApi.getClusterAddress();
    List<String> result = getResultListFromResponse(response, String.class);
    assertEquals(addr1, result.get(0));
    assertEquals(addr2, result.get(1));
  }

  @Test
  public void testGetClusterNodes() {
    HashMap<String, HashMap<String, Object>> clusterMetas = new HashMap<>();
    HashMap<String, Object> meta = new HashMap<>();
    String nodeName = "dummy";
    LocalDateTime SERVER_START_TIME = LocalDateTime.now();
    long cpuUsed = 20;
    long cpuCapacity = 40;
    long memoryUsed = 536870912;
    long memoryCapacity = 1073741824;
    meta.put(ClusterMeta.NODE_NAME, nodeName);
    meta.put(ClusterMeta.SERVER_START_TIME, SERVER_START_TIME);
    meta.put(ClusterMeta.CPU_USED, cpuUsed);
    meta.put(ClusterMeta.CPU_CAPACITY, cpuCapacity);
    meta.put(ClusterMeta.MEMORY_USED, memoryUsed);
    meta.put(ClusterMeta.MEMORY_CAPACITY, memoryCapacity);
    meta.put(ClusterMeta.STATUS, "OK");

    clusterMetas.put(nodeName, meta);
    mockClusterServer.putClusterMeta(SERVER_META, nodeName, meta);
    when(mockClusterServer.getClusterMeta(ClusterMetaType.SERVER_META, "")).thenReturn(clusterMetas);
    Response response = clusterRestApi.getClusterNodes();
    ArrayList<HashMap<String, Object>> result = getResultListFromResponse(response);
    Map<String, Object> properties = (LinkedTreeMap) result.get(0).get(ClusterMeta.PROPERTIES);

    assertEquals(clusterMetas.get(nodeName).get(ClusterMeta.NODE_NAME),
        result.get(0).get(ClusterMeta.NODE_NAME));
    assertEquals("OK", properties.get("STATUS"));
    assertEquals("0.50GB / 1.00GB = 50.00%", properties.get("MEMORY_USED / MEMORY_CAPACITY"));
    assertEquals("0.20 / 0.40 = 50.00%", properties.get("CPU_USED / CPU_CAPACITY"));
  }

  @Test
  public void testGetClusterNode() {
    HashMap<String, HashMap<String, Object>> clusterMetas = new HashMap<>();
    HashMap<String, Object> meta = new HashMap<>();
    String nodeName = "dummy";
    LocalDateTime SERVER_START_TIME = LocalDateTime.now();
    long cpuUsed = 20;
    long cpuCapacity = 40;
    long memoryUsed = 536870912;
    long memoryCapacity = 1073741824;
    meta.put(ClusterMeta.NODE_NAME, nodeName);
    meta.put(ClusterMeta.SERVER_START_TIME, SERVER_START_TIME);
    meta.put(ClusterMeta.CPU_USED, cpuUsed);
    meta.put(ClusterMeta.CPU_CAPACITY, cpuCapacity);
    meta.put(ClusterMeta.MEMORY_USED, memoryUsed);
    meta.put(ClusterMeta.MEMORY_CAPACITY, memoryCapacity);
    meta.put(ClusterMeta.STATUS, "OK");

    clusterMetas.put(nodeName, meta);
    mockClusterServer.putClusterMeta(SERVER_META, nodeName, meta);
    when(mockClusterServer.getClusterMeta(ClusterMetaType.SERVER_META, "")).thenReturn(clusterMetas);
    Response response = clusterRestApi.getClusterNode(nodeName,"");
    ArrayList<HashMap<String, Object>> result = getResultListFromResponse(response);
    Map<String, Object> properties = (LinkedTreeMap) result.get(0).get(ClusterMeta.PROPERTIES);

    assertEquals(clusterMetas.get(nodeName).get(ClusterMeta.NODE_NAME),
        result.get(0).get(ClusterMeta.NODE_NAME));
    assertEquals("OK", properties.get("STATUS"));
    assertEquals("0.50GB / 1.00GB = 50.00%", properties.get("MEMORY_USED / MEMORY_CAPACITY"));
    assertEquals("0.20 / 0.40 = 50.00%", properties.get("CPU_USED / CPU_CAPACITY"));
  }

  private <T> List<T> getResultListFromResponse(Response response, Class<T> typeT) {
    String entity = (String) response.getEntity();
    JsonObject object = new JsonParser().parse(entity).getAsJsonObject();
    JsonElement result = object.get("result");
    List<T> list = new ArrayList<T>();
    JsonArray array = result.getAsJsonArray();
    for (JsonElement jsonElement : array) {
      list.add(gson.fromJson(jsonElement, typeT));
    }
    return list;
  }

  private ArrayList<HashMap<String, Object>> getResultListFromResponse(Response response) {
    String entity = (String) response.getEntity();
    JsonObject object = new JsonParser().parse(entity).getAsJsonObject();
    JsonElement result = object.get("result");
    ArrayList<HashMap<String, Object>> list = new ArrayList<>();
    JsonArray array = result.getAsJsonArray();
    for (JsonElement jsonElement : array) {
      HashMap<String, Object> meta = new HashMap<>();
      meta = gson.fromJson(jsonElement, meta.getClass());
      list.add(meta);
    }
    return list;
  }
}
