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

import org.apache.submarine.commons.cluster.ClusterClient;
import org.apache.submarine.commons.cluster.meta.ClusterMetaType;
import org.apache.submarine.commons.utils.NetworkUtils;
import org.apache.submarine.commons.utils.SubmarineConfiguration;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.reflect.Constructor;
import java.util.HashMap;

import static java.lang.Thread.sleep;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

public class WorkbenchClusterServerTest {
  private static final Logger LOG = LoggerFactory.getLogger(WorkbenchClusterServerTest.class);

  private static ClusterClient clusterClient = null;

  @BeforeClass
  public static void start() throws Exception {
    LOG.info("WorkbenchClusterServerTest:start()");

    SubmarineConfiguration conf = SubmarineConfiguration.getInstance();
    String serverHost = NetworkUtils.findAvailableHostAddress();
    int serverPort = NetworkUtils.findRandomAvailablePortOnAllLocalInterfaces();
    String clusterAdd = serverHost + ":" + serverPort;
    conf.setClusterAddress(clusterAdd);

    // Run the workbench service in a thread
    AbstractWorkbenchServerTest.startUp(WorkbenchClusterServerTest.class.getSimpleName());

    // Mock Cluster client
    Class clazz = ClusterClient.class;
    Constructor constructor = null;
    constructor = clazz.getDeclaredConstructor();
    constructor.setAccessible(true);
    clusterClient = (ClusterClient) constructor.newInstance();
    clusterClient.start("TestWorkbenchClusterServer");

    // Waiting for cluster startup
    int wait = 0;
    while (wait++ < 100) {
      if (clusterClient.raftInitialized()) {
        LOG.info("TestWorkbenchClusterServer::start {}(ms) found cluster leader", wait * 3000);
        break;
      }

      sleep(3000);
    }

    assertTrue("Can not start Submarine workbench server!", clusterClient.raftInitialized());

    // Waiting for the workbench server to register in the cluster
    sleep(5000);
  }

  @AfterClass
  public static void stop() throws Exception {
    LOG.info("WorkbenchClusterServerTest::stop >>>");
    AbstractWorkbenchServerTest.shutDown();

    if (null != clusterClient) {
      clusterClient.shutdown();
    }
    LOG.info("WorkbenchClusterServerTest::stop <<<");
  }

  @Test
  public void testGetWorkbenchClusterMeta() {
    LOG.info("TestWorkbenchClusterServer::testGetWorkbenchClusterMeta >>>");
    // Get metadata for workbench server
    Object srvMeta = clusterClient.getClusterMeta(ClusterMetaType.SERVER_META, "");
    LOG.info("testGetWorkbenchClusterMeta = {}", srvMeta.toString());

    assertNotNull(srvMeta);
    assertEquals(true, (srvMeta instanceof HashMap));
    HashMap hashMap = (HashMap) srvMeta;

    assertEquals(hashMap.size(), 1);
    LOG.info("TestWorkbenchClusterServer::testGetWorkbenchClusterMeta <<<");
  }
}
