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
import java.time.format.DateTimeFormatter;
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

  private static HashMap<String, HashMap<String, Object>> clusterMetas = new HashMap<>();

  private static HashMap<String, Object> meta1 = new HashMap<>();
  private static HashMap<String, Object> meta2 = new HashMap<>();
  private static String nodeName1 = "dummy";
  private static LocalDateTime SERVER_START_TIME1 = LocalDateTime.now();
  private static LocalDateTime INTP_START_TIME = LocalDateTime.now();
  private static LocalDateTime LATEST_HEARTBEAT = LocalDateTime.now();
  private static long cpuUsed1 = 20;
  private static long cpuCapacity1 = 40;
  private static long memoryUsed1 = 536870912;
  private static long memoryCapacity1 = 1073741824;
  private static LocalDateTime SERVER_START_TIME2 = LocalDateTime.now();
  private static long cpuUsed2 = 25;
  private static long cpuCapacity2 = 40;
  private static long memoryUsed2 = 268435456;
  private static long memoryCapacity2 = 1073741824;
  private static String nodeName2 = "dummydummy";

  @BeforeClass
  public static void init() {
    mockClusterServer = mock(ClusterServer.class);
    clusterRestApi = new ClusterRestApi();
    clusterRestApi.setClusterServer(mockClusterServer);

    meta1.put(ClusterMeta.NODE_NAME, nodeName1);
    meta1.put(ClusterMeta.SERVER_START_TIME, SERVER_START_TIME1);
    meta1.put(ClusterMeta.CPU_USED, cpuUsed1);
    meta1.put(ClusterMeta.CPU_CAPACITY, cpuCapacity1);
    meta1.put(ClusterMeta.MEMORY_USED, memoryUsed1);
    meta1.put(ClusterMeta.MEMORY_CAPACITY, memoryCapacity1);
    meta1.put(ClusterMeta.INTP_START_TIME, INTP_START_TIME);
    meta1.put(ClusterMeta.LATEST_HEARTBEAT, LATEST_HEARTBEAT);
    meta1.put(ClusterMeta.STATUS, ClusterMeta.ONLINE_STATUS);

    meta2.put(ClusterMeta.NODE_NAME, nodeName2);
    meta2.put(ClusterMeta.SERVER_START_TIME, SERVER_START_TIME2);
    meta2.put(ClusterMeta.CPU_USED, cpuUsed2);
    meta2.put(ClusterMeta.CPU_CAPACITY, cpuCapacity2);
    meta2.put(ClusterMeta.MEMORY_USED, memoryUsed2);
    meta2.put(ClusterMeta.MEMORY_CAPACITY, memoryCapacity2);
    meta2.put(ClusterMeta.STATUS, ClusterMeta.OFFLINE_STATUS);

    clusterMetas.put(nodeName1, meta1);
    clusterMetas.put(nodeName2, meta2);
    mockClusterServer.putClusterMeta(SERVER_META, nodeName1, meta1);
    mockClusterServer.putClusterMeta(SERVER_META, nodeName2, meta2);
  }

  @Test
  public void testGetClusterAddress() {
    SubmarineConfiguration conf = SubmarineConfiguration.getInstance();
    String addr1 = "127.0.0.1:4569";
    String addr2 = "127.0.0.1:8888";
    conf.setClusterAddress(addr1 + "," + addr2);

    Response response = clusterRestApi.getClusterAddress();
    List<String> result = getResultListFromResponse(response, String.class);
    assertEquals(addr1, result.get(0));
    assertEquals(addr2, result.get(1));
  }

  @Test
  public void testGetClusterNodes() {
    when(mockClusterServer.getClusterMeta(ClusterMetaType.SERVER_META, "")).thenReturn(clusterMetas);
    Response response = clusterRestApi.getClusterNodes();
    ArrayList<HashMap<String, Object>> result = getResultListFromResponse(response);
    Map<String, Object> properties1 = (LinkedTreeMap) result.get(0).get(ClusterMeta.PROPERTIES);
    Map<String, Object> properties2 = (LinkedTreeMap) result.get(1).get(ClusterMeta.PROPERTIES);

    assertEquals(nodeName1, result.get(0).get(ClusterMeta.NODE_NAME));
    assertEquals("ONLINE", properties1.get("STATUS"));
    assertEquals("0.50GB / 1.00GB = 50.00%", properties1.get("MEMORY_USED / MEMORY_CAPACITY"));
    assertEquals("0.20 / 0.40 = 50.00%", properties1.get("CPU_USED / CPU_CAPACITY"));

    assertEquals(nodeName2, result.get(1).get(ClusterMeta.NODE_NAME));
    assertEquals("OFFLINE", properties2.get("STATUS"));
    assertEquals("0.25GB / 1.00GB = 25.00%", properties2.get("MEMORY_USED / MEMORY_CAPACITY"));
    assertEquals("0.25 / 0.40 = 62.50%", properties2.get("CPU_USED / CPU_CAPACITY"));
  }

  @Test
  public void testGetClusterNode() {
    when(mockClusterServer.getClusterMeta(ClusterMetaType.INTP_PROCESS_META, "")).thenReturn(clusterMetas);
    Response response = clusterRestApi.getClusterNode(nodeName1, "");
    ArrayList<HashMap<String, Object>> result = getResultListFromResponse(response);
    Map<String, Object> properties = (LinkedTreeMap) result.get(0).get(ClusterMeta.PROPERTIES);

    assertEquals(clusterMetas.get(nodeName1).get(ClusterMeta.NODE_NAME),
        result.get(0).get(ClusterMeta.NODE_NAME));
    assertEquals("ONLINE", properties.get("STATUS"));
    assertEquals(INTP_START_TIME.format(DateTimeFormatter.ISO_DATE_TIME), properties.get("INTP_START_TIME"));
    assertEquals(LATEST_HEARTBEAT.format(DateTimeFormatter.ISO_DATE_TIME),
        properties.get("LATEST_HEARTBEAT"));
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
