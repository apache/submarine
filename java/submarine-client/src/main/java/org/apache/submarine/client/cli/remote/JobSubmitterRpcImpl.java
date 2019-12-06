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

package org.apache.submarine.client.cli.remote;

import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import io.grpc.StatusRuntimeException;
import org.apache.hadoop.yarn.api.records.ApplicationId;
import org.apache.hadoop.yarn.exceptions.YarnException;
import org.apache.submarine.commons.rpc.ApplicationIdProto;
import org.apache.submarine.commons.rpc.ParameterProto;
import org.apache.submarine.commons.rpc.SubmarineServerProtocolGrpc;
import org.apache.submarine.commons.rpc.SubmarineServerProtocolGrpc.SubmarineServerProtocolBlockingStub;
import org.apache.submarine.commons.rpc.SubmarineServerProtocolGrpc.SubmarineServerProtocolStub;
import org.apache.submarine.commons.runtime.ClientContext;
import org.apache.submarine.commons.runtime.JobSubmitter;
import org.apache.submarine.commons.runtime.param.Parameter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.concurrent.TimeUnit;

public class JobSubmitterRpcImpl implements JobSubmitter {
  private static final Logger LOG =
      LoggerFactory.getLogger(JobSubmitterRpcImpl.class.getName());

  private final ManagedChannel channel;
  private final SubmarineServerProtocolBlockingStub blockingStub;
  private final SubmarineServerProtocolStub asyncStub;
  private final RpcContext rpcContext;

  /** Construct client for accessing RouteGuide server at {@code host:port}. */
  public JobSubmitterRpcImpl(String host, int port,
      ClientContext clientContext) {
    this(ManagedChannelBuilder.forAddress(host, port).usePlaintext(),
        clientContext);
  }

  /** Construct client for accessing RouteGuide server using the existing channel. */
  public JobSubmitterRpcImpl(ManagedChannelBuilder<?> channelBuilder,
      ClientContext clientContext) {
    channel = channelBuilder.build();
    blockingStub = SubmarineServerProtocolGrpc.newBlockingStub(channel);
    asyncStub = SubmarineServerProtocolGrpc.newStub(channel);
    rpcContext = RpcContext.convertClientContextToRpcContext(clientContext);
  }

  @Override
  public ApplicationId submitJob(Parameter parameters) throws IOException, YarnException {
    ParameterProto request = ClientProto.convertParameterToParameterProto(
        parameters, rpcContext);

    ApplicationId applicationId = null;
    try {
      ApplicationIdProto applicationIdProto = blockingStub.submitJob(request);
      applicationId =
          ClientProto.convertApplicationIdProtoToApplicationId(applicationIdProto);
    } catch (StatusRuntimeException e) {
      LOG.error(e.getMessage(),e);
    } finally {
      shutdown();
    }
    return applicationId;
  }

  public void shutdown() {
    try {
      channel.shutdown().awaitTermination(5, TimeUnit.SECONDS);
    } catch (InterruptedException e) {
      LOG.error(e.getMessage(), e);
    }
  }

}
