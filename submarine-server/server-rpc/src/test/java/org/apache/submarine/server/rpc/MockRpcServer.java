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

import io.grpc.ServerBuilder;
import org.apache.hadoop.yarn.api.records.ApplicationId;
import org.apache.submarine.client.cli.CliUtils;
import org.apache.submarine.commons.runtime.ClientContext;
import org.apache.submarine.commons.runtime.param.Parameter;
import org.apache.submarine.commons.utils.SubmarineConfiguration;

import java.io.IOException;

public class MockRpcServer extends SubmarineRpcServer {

  public MockRpcServer(int port) throws IOException {
    this(ServerBuilder.forPort(port), port);
  }

  public MockRpcServer(ServerBuilder<?> serverBuilder, int port) {
    super(serverBuilder, port, new mockSubmarineServerRpcService());
  }

  protected static class mockSubmarineServerRpcService
      extends SubmarineServerRpcService {
    @Override
    protected ApplicationId run(ClientContext clientContext,
        Parameter parameter) {
      return CliUtils.fromString("application_1_123");
    }
  }

  public static void main(String[] args) throws Exception {
    SubmarineRpcServer server = startRpcServer();
    server.blockUntilShutdown();
  }

  public static SubmarineRpcServer startRpcServer() throws IOException {
    SubmarineConfiguration submarineConfiguration =
        SubmarineConfiguration.getInstance();
    int rpcServerPort = submarineConfiguration.getInt(
        SubmarineConfiguration.ConfVars.SUBMARINE_SERVER_REMOTE_EXECUTION_PORT);
    SubmarineRpcServer server = new MockRpcServer(rpcServerPort);
    server.start();
    return server;
  }

}