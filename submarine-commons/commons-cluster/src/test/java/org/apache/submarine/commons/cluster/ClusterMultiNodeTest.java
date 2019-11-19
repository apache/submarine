/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.submarine.commons.cluster;

import org.apache.submarine.commons.cluster.meta.ClusterMetaType;
import org.apache.submarine.commons.utils.SubmarineConfiguration;
import org.apache.submarine.commons.utils.NetworkUtils;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

public class ClusterMultiNodeTest {
  private static Logger LOG = LoggerFactory.getLogger(ClusterMultiNodeTest.class);

  private static List<ClusterServer> clusterServers = new ArrayList<>();
  private static ClusterClient clusterClient = null;

  static final String metaKey = "ClusterMultiNodeTestKey";

  @BeforeClass
  public static void startCluster() throws IOException, InterruptedException {
    LOG.info("ClusterMultiNodeTest::startCluster >>>");

    String clusterAddrList = "";
    String serverHost = NetworkUtils.findAvailableHostAddress();
    for (int i = 0; i < 3; i++) {
      // Set the cluster IP and port
      int serverPort = NetworkUtils.findRandomAvailablePortOnAllLocalInterfaces();
      clusterAddrList += serverHost + ":" + serverPort;
      if (i != 2) {
        clusterAddrList += ",";
      }
    }
    LOG.info("clusterAddrList = {}", clusterAddrList);
    SubmarineConfiguration sconf = SubmarineConfiguration.getInstance();
    sconf.setClusterAddress(clusterAddrList);

    // mock cluster manager server
    String cluster[] = clusterAddrList.split(",");
    try {
      for (int i = 0; i < 3; i++) {
        String[] parts = cluster[i].split(":");
        String clusterHost = parts[0];
        int clusterPort = Integer.valueOf(parts[1]);

        Class clazz = ClusterServer.class;
        Constructor constructor = clazz.getDeclaredConstructor();
        constructor.setAccessible(true);
        ClusterServer clusterServer = (ClusterServer) constructor.newInstance();
        clusterServer.initTestCluster(clusterAddrList, clusterHost, clusterPort);

        clusterServers.add(clusterServer);
      }
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
    }

    for (ClusterServer clusterServer : clusterServers) {
      clusterServer.start();
    }

    // mock cluster manager client
    try {
      Class clazz = ClusterClient.class;
      Constructor constructor = null;
      constructor = clazz.getDeclaredConstructor();
      constructor.setAccessible(true);
      clusterClient = (ClusterClient) constructor.newInstance();
      clusterClient.start(metaKey);
    } catch (NoSuchMethodException | InstantiationException
        | IllegalAccessException | InvocationTargetException e) {
      LOG.error(e.getMessage(), e);
    }

    // Waiting for cluster startup
    boolean clusterIsStartup = false;
    int wait = 0;
    while (wait++ < 100) {
      if (clusterIsStartup() && clusterClient.raftInitialized()) {
        LOG.info("ClusterMultiNodeTest::wait {}(ms) found cluster leader", wait * 3000);
        clusterIsStartup = true;
        break;
      }
      try {
        Thread.sleep(3000);
      } catch (InterruptedException e) {
        LOG.error(e.getMessage(), e);
      }
    }

    assertEquals(clusterIsStartup, true);

    Thread.sleep(5000);
    assertEquals(true, clusterIsStartup());
    LOG.info("ClusterMultiNodeTest::startCluster <<<");
  }

  @AfterClass
  public static void stopCluster() {
    LOG.info("ClusterMultiNodeTest::stopCluster >>>");
    if (null != clusterClient) {
      clusterClient.shutdown();
    }
    for (ClusterServer clusterServer : clusterServers) {
      clusterServer.shutdown();
    }
    LOG.info("ClusterMultiNodeTest::stopCluster <<<");
  }

  static boolean clusterIsStartup() {
    boolean foundLeader = false;
    for (ClusterServer clusterServer : clusterServers) {
      if (!clusterServer.raftInitialized()) {
        LOG.warn("clusterServer not Initialized!");
        return false;
      }
      if (clusterServer.isClusterLeader()) {
        foundLeader = true;
      }
    }

    if (!foundLeader) {
      LOG.warn("Can not found leader!");
      return false;
    }

    LOG.info("cluster startup!");
    return true;
  }

  @Test
  public void testClusterServerMeta() {
    LOG.info("ClusterMultiNodeTest::testClusterServerMeta >>>");
    // Get metadata for all services
    Object srvMeta = clusterClient.getClusterMeta(ClusterMetaType.SERVER_META, "");
    LOG.info(srvMeta.toString());

    assertNotNull(srvMeta);
    assertEquals(true, (srvMeta instanceof HashMap));
    HashMap hashMap = (HashMap) srvMeta;

    assertEquals(hashMap.size(), 3);
    LOG.info("ClusterMultiNodeTest::testClusterServerMeta <<<");
  }

  @Test
  public void testClusterClientMeta() {
    LOG.info("ClusterMultiNodeTest::testClusterClientMeta >>>");
    // Get metadata for all services
    Object srvMeta = clusterClient.getClusterMeta(ClusterMetaType.INTP_PROCESS_META, "");
    LOG.info(srvMeta.toString());

    assertNotNull(srvMeta);
    assertEquals(true, (srvMeta instanceof HashMap));
    HashMap hashMap = (HashMap) srvMeta;

    assertEquals(hashMap.size(), 1);
    LOG.info("ClusterMultiNodeTest::testClusterClientMeta <<<");
  }
}
