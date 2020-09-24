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
import org.apache.submarine.commons.cluster.meta.ClusterMeta;
import org.apache.submarine.commons.cluster.meta.ClusterMetaType;
import org.apache.submarine.commons.utils.NetworkUtils;
import org.apache.submarine.commons.utils.SubmarineConfiguration;
import org.apache.submarine.server.SubmarineServer;
import org.apache.submarine.server.api.experiment.Experiment;
import org.apache.submarine.server.api.experiment.ExperimentId;
import org.apache.submarine.server.api.experiment.ExperimentLog;
import org.apache.submarine.server.api.spec.EnvironmentSpec;
import org.apache.submarine.server.api.spec.ExperimentMeta;
import org.apache.submarine.server.api.spec.ExperimentSpec;
import org.apache.submarine.server.api.spec.KernelSpec;
import org.apache.submarine.server.experiment.ExperimentManager;
import org.apache.submarine.server.gson.ExperimentIdDeserializer;
import org.apache.submarine.server.gson.ExperimentIdSerializer;
import org.apache.submarine.commons.cluster.ClusterServer;
import org.junit.Test;
import org.junit.BeforeClass;
import org.junit.Before;

import javax.ws.rs.core.Response;
import java.time.LocalDateTime;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

import static org.apache.submarine.commons.cluster.meta.ClusterMetaType.SERVER_META;
import static org.junit.Assert.assertEquals;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
public class ClusterRestApiTest {
  private static ClusterServer mockClusterServer;
  private static ClusterRestApi clusterRestApi;

  private static final GsonBuilder gsonBuilder = new GsonBuilder();

  private static Gson gson = gsonBuilder.setDateFormat("yyyy-MM-dd HH:mm:ss").create();

  private static String addr1="0.1.2.3:4569";
  private static String addr2="4.5.6.7:8888";
  private static String dummyHost="1.3.5.7";
  private static int dummyPort =2468;

  @BeforeClass
  public static void init() {
    mockClusterServer = mock(ClusterServer.class);

    //mockClusterServer.initTestCluster((addr1+","+addr2),dummyHost,dummyPort);
    clusterRestApi = new ClusterRestApi();
    clusterRestApi.setClusterServer(mockClusterServer);
  }

  @Test
  public void testGetClusterAddress(){
    SubmarineConfiguration conf = SubmarineConfiguration.getInstance();
    conf.setClusterAddress(addr1+","+addr2);

    Response response = clusterRestApi.getClusterAddress();
    List<String> result = getResultListFromResponse(response, String.class);
    assertEquals(addr1, result.get(0));
    assertEquals(addr2, result.get(1));

  }

  @Test
  public void testGetClusterNodes(){
    //HashMap<String, HashMap<String, Object>> clusterMeta;
    HashMap<String, Object> meta = new HashMap<String, Object>();
    String nodeName = "node name";
    meta.put(ClusterMeta.NODE_NAME, nodeName);
    meta.put(ClusterMeta.SERVER_HOST, dummyHost);
    meta.put(ClusterMeta.SERVER_PORT, dummyPort);
    meta.put(ClusterMeta.SERVER_START_TIME, LocalDateTime.now());
    mockClusterServer.putClusterMeta(SERVER_META, nodeName, meta);
    when(mockClusterServer.getClusterMeta(ClusterMetaType.SERVER_META,"")).thenReturn(meta);
    Response response = clusterRestApi.getClusterNodes();

    System.out.print("");
  }

  @Test
  public void testGetClusterNode(){

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
}
