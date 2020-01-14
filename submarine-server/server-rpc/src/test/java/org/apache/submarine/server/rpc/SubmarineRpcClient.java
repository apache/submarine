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

package org.apache.submarine.server.rpc;

import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import io.grpc.StatusRuntimeException;
import org.apache.submarine.commons.rpc.ApplicationIdProto;
import org.apache.submarine.commons.rpc.ParametersHolderProto;
import org.apache.submarine.commons.rpc.SubmarineServerProtocolGrpc;
import org.apache.submarine.commons.rpc.SubmarineServerProtocolGrpc.SubmarineServerProtocolBlockingStub;
import org.apache.submarine.commons.rpc.SubmarineServerProtocolGrpc.SubmarineServerProtocolStub;
import org.apache.submarine.commons.utils.SubmarineConfiguration;
import org.apache.submarine.commons.utils.SubmarineConfVars;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.TimeUnit;


/**
 * Sample client code that makes gRPC calls to the server.
 */
public class SubmarineRpcClient extends RpcServerTestUtils {
  private static final Logger LOG =
      LoggerFactory.getLogger(SubmarineRpcClient.class.getName());

  protected final ManagedChannel channel;
  protected final SubmarineServerProtocolBlockingStub blockingStub;
  protected final SubmarineServerProtocolStub asyncStub;

  public SubmarineRpcClient(SubmarineConfiguration config) {
    this(ManagedChannelBuilder.forAddress(
          config.getString(
            SubmarineConfVars.ConfVars.SUBMARINE_SERVER_ADDR),
          config.getInt(
            SubmarineConfVars.ConfVars.
                SUBMARINE_SERVER_REMOTE_EXECUTION_PORT))
        .usePlaintext());
  }

  /** Construct client for accessing RouteGuide server at {@code host:port}. */
  public SubmarineRpcClient(String host, int port) {
    this(ManagedChannelBuilder.forAddress(host, port).usePlaintext());
  }

  /** Construct client for accessing RouteGuide server using the existing channel. */
  public SubmarineRpcClient(ManagedChannelBuilder<?> channelBuilder) {
    channel = channelBuilder.build();
    blockingStub = SubmarineServerProtocolGrpc.newBlockingStub(channel);
    asyncStub = SubmarineServerProtocolGrpc.newStub(channel);
  }

  public void shutdown() throws InterruptedException {
    channel.shutdown().awaitTermination(5, TimeUnit.SECONDS);
  }

  public boolean testRpcConnection() throws InterruptedException {
    LOG.info("Try to connect to submarine rpc server.");
    boolean isRunning = false;

    ParametersHolderProto request =
        ParametersHolderProto.newBuilder().setHelloworld(1).build();
    try {
      blockingStub.testRpc(request);
      isRunning = true;
    } catch (StatusRuntimeException e) {
      LOG.error(e.getMessage(),e);
    } finally {
      shutdown();
    }
    return isRunning;
  }


  public static void main(String[] args) throws InterruptedException {
    SubmarineRpcClient client = new SubmarineRpcClient("localhost", 8980);
    try {
      // Looking for a valid feature
      client.testRpcConnection();
    } finally {
      client.shutdown();
    }
  }

}
