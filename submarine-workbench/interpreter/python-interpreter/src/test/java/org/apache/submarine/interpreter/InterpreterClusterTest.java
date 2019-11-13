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
package org.apache.submarine.interpreter;

import org.apache.submarine.commons.cluster.ClusterClient;
import org.apache.submarine.commons.cluster.ClusterServer;
import org.apache.submarine.commons.utils.NetworkUtils;
import org.apache.submarine.commons.utils.SubmarineConfiguration;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.util.HashMap;

import static org.apache.submarine.commons.cluster.meta.ClusterMetaType.INTP_PROCESS_META;
import static org.apache.submarine.commons.cluster.meta.ClusterMetaType.SERVER_META;
import static org.junit.Assert.assertEquals;

public class InterpreterClusterTest {
  private static Logger LOG = LoggerFactory.getLogger(InterpreterClusterTest.class);
  private static SubmarineConfiguration sconf;

  private static ClusterServer clusterServer = null;
  private static ClusterClient clusterClient = null;

  static String serverHost;
  static int serverPort;
  static final String clientMetaKey = "InterpreterProcessTest";

  @BeforeClass
  public static void startCluster() throws IOException, InterruptedException {
    LOG.info("startCluster >>>");

    sconf = SubmarineConfiguration.getInstance();

    // Set the cluster IP and port
    serverHost = NetworkUtils.findAvailableHostAddress();
    serverPort = NetworkUtils.findRandomAvailablePortOnAllLocalInterfaces();
    sconf.setClusterAddress(serverHost + ":" + serverPort);

    // mock cluster manager server
    clusterServer = ClusterServer.getInstance();
    clusterServer.start();

    // mock cluster manager client
    try {
      Class clazz = ClusterClient.class;
      Constructor constructor = null;
      constructor = clazz.getDeclaredConstructor();
      constructor.setAccessible(true);
      clusterClient = (ClusterClient) constructor.newInstance();
      clusterClient.start(clientMetaKey);
    } catch (NoSuchMethodException | InstantiationException
        | IllegalAccessException | InvocationTargetException e) {
      LOG.error(e.getMessage(), e);
    }

    // Waiting for cluster startup
    int wait = 0;
    while (wait++ < 100) {
      if (clusterServer.isClusterLeader()
          && clusterServer.raftInitialized()
          && clusterClient.raftInitialized()) {
        LOG.info("wait {}(ms) found cluster leader", wait * 3000);
        break;
      }
      Thread.sleep(3000);
    }
    Thread.sleep(3000);
    assertEquals(true, clusterServer.isClusterLeader()
        && clusterServer.raftInitialized()
        && clusterClient.raftInitialized());
    LOG.info("startCluster <<<");
  }

  @AfterClass
  public static void stopCluster() {
    if (null != clusterClient) {
      clusterClient.shutdown();
    }
    if (null != clusterClient) {
      clusterServer.shutdown();
    }
    LOG.info("stopCluster");
  }

  @Test
  public void testInterpreterProcess() throws IOException, InterruptedException {
    InterpreterProcess interpreterProcess = new InterpreterProcess("python", "testInterpreterProcess", false);

    startInterpreterProcess(interpreterProcess, 5000);

    // Get metadata for all services
    HashMap<String, HashMap<String, Object>> serverMeta
        = clusterClient.getClusterMeta(SERVER_META, clusterClient.getClusterNodeName());
    LOG.info("serverMeta.size = {}", serverMeta.size());
    assertEquals(serverMeta.size(), 1);

    // get IntpProcess Meta
    HashMap<String, HashMap<String, Object>> intpMeta
        = clusterClient.getClusterMeta(INTP_PROCESS_META, "testInterpreterProcess");
    LOG.info("intpMeta.size = {}", intpMeta.size());
    assertEquals(intpMeta.size(), 1);

    HashMap<String, HashMap<String, Object>> intpMeta2
        = clusterClient.getClusterMeta(INTP_PROCESS_META, clientMetaKey);
    LOG.info("intpMeta2.size = {}", intpMeta2.size());
    assertEquals(intpMeta2.size(), 1);

    stopInterpreterProcess(interpreterProcess, 5000);
  }

  private void startInterpreterProcess(InterpreterProcess interpreterProcess, int timeout)
      throws InterruptedException, IOException {
    interpreterProcess.start();
    long startTime = System.currentTimeMillis();
    while (System.currentTimeMillis() - startTime < timeout) {
      if (interpreterProcess.isRunning()) {
        break;
      }
      Thread.sleep(200);
    }
    assertEquals(true, interpreterProcess.isRunning());
  }

  private void stopInterpreterProcess(InterpreterProcess interpreterProcess, int timeout)
      throws InterruptedException {
    interpreterProcess.shutdown();
    long startTime = System.currentTimeMillis();
    while (System.currentTimeMillis() - startTime < timeout) {
      if (!interpreterProcess.isRunning()) {
        break;
      }
      Thread.sleep(200);
    }
    assertEquals(false, interpreterProcess.isRunning());
  }
}
