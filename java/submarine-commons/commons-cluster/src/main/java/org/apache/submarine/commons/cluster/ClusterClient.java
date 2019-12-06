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

import io.atomix.primitive.PrimitiveState;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static org.apache.submarine.commons.cluster.meta.ClusterMetaType.INTP_PROCESS_META;

/**
 * Cluster management client class instantiated in submarine-interperter
 */
public class ClusterClient extends ClusterManager {
  private static Logger LOG = LoggerFactory.getLogger(ClusterClient.class);

  private static ClusterClient instance = null;

  // Do not use the getInstance function in the test case,
  // which will result in an inability to update the instance according to the configuration.
  public static ClusterClient getInstance() {
    synchronized (ClusterClient.class) {
      if (instance == null) {
        instance = new ClusterClient();
      }
    }
    return instance;
  }

  private ClusterClient() {
    super();
  }

  @Override
  public boolean raftInitialized() {
    if (null != raftClient && null != raftSessionClient
        && raftSessionClient.getState() == PrimitiveState.CONNECTED) {
      return true;
    }

    return false;
  }

  @Override
  public boolean isClusterLeader() {
    return false;
  }

  // In the ClusterClient metaKey equal interperterGroupId
  public void start(String metaKey) {
    LOG.info("ClusterClient::start({})", metaKey);
    if (!sconf.isClusterMode()) {
      return;
    }
    super.start();

    // Instantiated cluster monitoring class
    clusterMonitor = new ClusterMonitor(this);
    clusterMonitor.start(INTP_PROCESS_META, metaKey);
  }

  public void shutdown() {
    if (!sconf.isClusterMode()) {
      return;
    }
    if (null != clusterMonitor) {
      clusterMonitor.shutdown();
    }

    super.shutdown();
  }
}
