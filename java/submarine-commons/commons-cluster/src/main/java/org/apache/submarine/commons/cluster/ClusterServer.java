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

import com.google.common.annotations.VisibleForTesting;
import io.atomix.cluster.BootstrapService;
import io.atomix.cluster.ManagedClusterMembershipService;
import io.atomix.cluster.Member;
import io.atomix.cluster.MemberId;
import io.atomix.cluster.MembershipConfig;
import io.atomix.cluster.Node;
import io.atomix.cluster.discovery.BootstrapDiscoveryProvider;
import io.atomix.cluster.impl.DefaultClusterMembershipService;
import io.atomix.cluster.impl.DefaultNodeDiscoveryService;
import io.atomix.cluster.messaging.BroadcastService;
import io.atomix.cluster.messaging.MessagingService;
import io.atomix.cluster.messaging.impl.NettyMessagingService;
import io.atomix.primitive.PrimitiveState;
import io.atomix.protocols.raft.RaftServer;
import io.atomix.protocols.raft.protocol.RaftServerProtocol;
import io.atomix.protocols.raft.storage.RaftStorage;
import io.atomix.storage.StorageLevel;
import io.atomix.utils.net.Address;
import org.apache.commons.lang.StringUtils;
import org.apache.submarine.commons.cluster.meta.ClusterMeta;
import org.apache.submarine.commons.cluster.protocol.RaftServerMessagingProtocol;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.time.Duration;
import java.time.LocalDateTime;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

import static org.apache.submarine.commons.cluster.meta.ClusterMetaType.SERVER_META;

/**
 * Cluster management server class instantiated in submarine-server
 * 1. Create a raft server
 * 2. Remotely create interpreter's thrift service
 */
public class ClusterServer extends ClusterManager {
  private static Logger LOG = LoggerFactory.getLogger(ClusterServer.class);

  private static ClusterServer instance = null;

  // raft server
  protected RaftServer raftServer = null;

  protected MessagingService messagingService = null;

  private ClusterServer() {
    super();
  }

  // Do not use the getInstance function in the test case,
  // which will result in an inability to update the instance according to the configuration.
  public static ClusterServer getInstance() {
    synchronized (ClusterServer.class) {
      if (instance == null) {
        instance = new ClusterServer();
      }
    }
    return instance;
  }

  public void start() {
    if (!sconf.isClusterMode()) {
      return;
    }

    initThread();

    // Instantiated raftServer monitoring class
    String clusterName = getClusterNodeName();
    clusterMonitor = new ClusterMonitor(this);
    clusterMonitor.start(SERVER_META, clusterName);

    super.start();
  }

  @VisibleForTesting
  void initTestCluster(String clusterAddrList, String host, int port) {
    isTest = true;
    this.serverHost = host;
    this.raftServerPort = port;

    // clear
    clusterNodes.clear();
    raftAddressMap.clear();
    clusterMemberIds.clear();

    String cluster[] = clusterAddrList.split(",");
    for (int i = 0; i < cluster.length; i++) {
      String[] parts = cluster[i].split(":");
      String clusterHost = parts[0];
      int clusterPort = Integer.valueOf(parts[1]);

      String memberId = clusterHost + ":" + clusterPort;
      Address address = Address.from(clusterHost, clusterPort);
      Node node = Node.builder().withId(memberId).withAddress(address).build();
      clusterNodes.add(node);
      raftAddressMap.put(MemberId.from(memberId), address);
      clusterMemberIds.add(MemberId.from(memberId));
    }
  }

  @Override
  public boolean raftInitialized() {
    if (null != raftServer && raftServer.isRunning()
        && null != raftClient && null != raftSessionClient
        && raftSessionClient.getState() == PrimitiveState.CONNECTED) {
      return true;
    }

    return false;
  }

  @Override
  public boolean isClusterLeader() {
    if (null == raftServer
        || !raftServer.isRunning()
        || !raftServer.isLeader()) {
      return false;
    }

    return true;
  }

  private void initThread() {
    // RaftServer Thread
    new Thread(new Runnable() {
      @Override
      public void run() {
        LOG.info("RaftServer run() >>>");

        Address address = Address.from(serverHost, raftServerPort);
        Member member = Member.builder(MemberId.from(serverHost + ":" + raftServerPort))
            .withAddress(address)
            .build();
        messagingService = NettyMessagingService.builder()
            .withAddress(address).build().start().join();
        RaftServerProtocol protocol = new RaftServerMessagingProtocol(
            messagingService, ClusterManager.protocolSerializer, raftAddressMap::get);

        BootstrapService bootstrapService = new BootstrapService() {
          @Override
          public MessagingService getMessagingService() {
            return messagingService;
          }

          @Override
          public BroadcastService getBroadcastService() {
            return new BroadcastServiceAdapter();
          }
        };

        ManagedClusterMembershipService clusterService = new DefaultClusterMembershipService(
            member,
            new DefaultNodeDiscoveryService(bootstrapService, member,
                new BootstrapDiscoveryProvider(clusterNodes)),
            bootstrapService,
            new MembershipConfig());

        File atomixDateDir = com.google.common.io.Files.createTempDir();
        atomixDateDir.deleteOnExit();

        RaftServer.Builder builder = RaftServer.builder(member.id())
            .withMembershipService(clusterService)
            .withProtocol(protocol)
            .withStorage(RaftStorage.builder()
                .withStorageLevel(StorageLevel.MEMORY)
                .withDirectory(atomixDateDir)
                .withSerializer(storageSerializer)
                .withMaxSegmentSize(1024 * 1024)
                .build());

        raftServer = builder.build();
        raftServer.bootstrap(clusterMemberIds);

        HashMap<String, Object> meta = new HashMap<String, Object>();
        String nodeName = getClusterNodeName();
        meta.put(ClusterMeta.NODE_NAME, nodeName);
        meta.put(ClusterMeta.SERVER_HOST, serverHost);
        meta.put(ClusterMeta.SERVER_PORT, raftServerPort);
        meta.put(ClusterMeta.SERVER_START_TIME, LocalDateTime.now());
        putClusterMeta(SERVER_META, nodeName, meta);

        LOG.info("RaftServer run() <<<");
      }
    }).start();
  }

  @Override
  public void shutdown() {
    if (!sconf.isClusterMode()) {
      return;
    }
    LOG.info("ClusterServer::shutdown()");

    try {
      // delete local machine meta
      deleteClusterMeta(SERVER_META, getClusterNodeName());
      Thread.sleep(500);
      if (null != clusterMonitor) {
        clusterMonitor.shutdown();
      }
      // wait raft commit metadata
      Thread.sleep(500);
    } catch (InterruptedException e) {
      LOG.error(e.getMessage(), e);
    }

    // close raft client
    super.shutdown();

    if (null != raftServer && raftServer.isRunning()) {
      try {
        LOG.info("ClusterServer::raftServer.shutdown()");
        raftServer.shutdown().get(5, TimeUnit.SECONDS);
      } catch (InterruptedException | ExecutionException | TimeoutException e) {
        LOG.error(e.getMessage(), e);
      }
    }

    LOG.info("ClusterServer::super.shutdown()");
  }

  // Obtain the server node whose resources are idle in the cluster
  public HashMap<String, Object> getIdleNodeMeta() {
    HashMap<String, Object> idleNodeMeta = null;
    HashMap<String, HashMap<String, Object>> clusterMeta = getClusterMeta(SERVER_META, "");

    long memoryIdle = 0;
    for (Map.Entry<String, HashMap<String, Object>> entry : clusterMeta.entrySet()) {
      HashMap<String, Object> meta = entry.getValue();
      // Check if the service or process is offline
      String status = (String) meta.get(ClusterMeta.STATUS);
      if (null == status || StringUtils.isEmpty(status)
          || status.equals(ClusterMeta.OFFLINE_STATUS)) {
        continue;
      }

      long memoryCapacity  = (long) meta.get(ClusterMeta.MEMORY_CAPACITY);
      long memoryUsed      = (long) meta.get(ClusterMeta.MEMORY_USED);
      long idle = memoryCapacity - memoryUsed;
      if (idle > memoryIdle) {
        memoryIdle = idle;
        idleNodeMeta = meta;
      }
    }

    return idleNodeMeta;
  }

  public void unicastClusterEvent(String host, int port, String topic, String msg) {
    LOG.info("send unicastClusterEvent host:{} port:{} topic:{} message:{}",
        host, port, topic, msg);

    Address address = Address.from(host, port);
    CompletableFuture<byte[]> response = messagingService.sendAndReceive(address,
        topic, msg.getBytes(), Duration.ofSeconds(2));
    response.whenComplete((r, e) -> {
      if (null == e) {
        LOG.error(e.getMessage(), e);
      }
    });
  }

  public void broadcastClusterEvent(String topic, String msg) {
    if (LOG.isDebugEnabled()) {
      LOG.debug("send broadcastClusterEvent message {}", msg);
    }
    for (Node node : clusterNodes) {
      if (StringUtils.equals(node.address().host(), serverHost)
          && node.address().port() == raftServerPort) {
        // skip myself
        continue;
      }

      CompletableFuture<byte[]> response = messagingService.sendAndReceive(node.address(),
          topic, msg.getBytes(), Duration.ofSeconds(2));
      response.whenComplete((r, e) -> {
        if (null == e) {
          LOG.error(e.getMessage(), e);
        } else {
          LOG.info("broadcastClusterNoteEvent success! {}", msg);
        }
      });
    }
  }
}
