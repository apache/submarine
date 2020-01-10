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

package org.apache.submarine.server.rpc;

import io.grpc.Server;
import io.grpc.ServerBuilder;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.yarn.api.records.ApplicationId;
import org.apache.hadoop.yarn.conf.YarnConfiguration;
import org.apache.hadoop.yarn.exceptions.YarnException;
import org.apache.submarine.client.cli.remote.RpcContext;
import org.apache.submarine.commons.rpc.ApplicationIdProto;
import org.apache.submarine.commons.rpc.ParameterProto;
import org.apache.submarine.commons.rpc.ParametersHolderProto;
import org.apache.submarine.commons.rpc.SubmarineServerProtocolGrpc;
import org.apache.submarine.commons.runtime.ClientContext;
import org.apache.submarine.commons.runtime.JobSubmitter;
import org.apache.submarine.commons.runtime.exception.SubmarineException;
import org.apache.submarine.commons.runtime.RuntimeFactory;
import org.apache.submarine.commons.runtime.param.Parameter;
import org.apache.submarine.commons.utils.SubmarineConfiguration;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.Map;

/**
 * A gRPC server that provides submarine service.
 */
public class SubmarineRpcServer {
  private static final Logger LOG = LoggerFactory.getLogger(
      SubmarineRpcServer.class.getName());

  protected int port;
  protected Server server;

  public SubmarineRpcServer(int port) throws IOException {
    this(ServerBuilder.forPort(port), port);
  }

  /** Create a RouteGuide server using serverBuilder as a base and features as data. */
  public SubmarineRpcServer(ServerBuilder<?> serverBuilder, int port) {
    this(serverBuilder, port, new SubmarineServerRpcService());
  }

  public SubmarineRpcServer(int port,
      SubmarineServerProtocolGrpc.SubmarineServerProtocolImplBase service) {
    this(ServerBuilder.forPort(port), port, service);
  }

  public SubmarineRpcServer(ServerBuilder<?> serverBuilder, int port,
      SubmarineServerProtocolGrpc.SubmarineServerProtocolImplBase service) {
    this.port = port;
    server = serverBuilder.addService(service).build();
  }

  /** Start serving requests. */
  public void start() throws IOException {
    server.start();
    LOG.info("Server started, listening on " + port);
    Runtime.getRuntime().addShutdownHook(new Thread() {
      @Override
      public void run() {
        // Use stderr here since the logger may have been reset by its JVM shutdown hook.
        LOG.info("*** shutting down gRPC server since JVM is shutting down");
        SubmarineRpcServer.this.stop();
      }
    });
  }

  /** Stop serving requests and shutdown resources. */
  public void stop() {
    if (server != null) {
      server.shutdown();
      LOG.info("*** server shut down");
    }
  }

  /**
   * Await termination on the main thread since the grpc library uses daemon threads.
   */
  public void blockUntilShutdown() throws InterruptedException {
    if (server != null) {
      server.awaitTermination();
    }
  }

  private static ClientContext getClientContext(RpcContext rpcContext) {
    Configuration conf = new YarnConfiguration();
    ClientContext clientContext = new ClientContext();
    clientContext.setYarnConfig(conf);
    mergeSubmarineConfiguration(clientContext.getSubmarineConfig(), rpcContext);
    RuntimeFactory runtimeFactory = RuntimeFactory.getRuntimeFactory(
        clientContext);
    clientContext.setRuntimeFactory(runtimeFactory);
    return clientContext;
  }

  private static void mergeSubmarineConfiguration(
      SubmarineConfiguration submarineConfiguration, RpcContext rpcContext) {
    Map<String, String> submarineJobConfigMap =
        rpcContext.getSubmarineJobConfigMap();
    for(Map.Entry<String, String> entry: submarineJobConfigMap.entrySet()){
      submarineConfiguration.updateConfiguration(
          entry.getKey(), entry.getValue());
    }
  }

  /**
   * Main method.  This comment makes the linter happy.
   */
  public static void main(String[] args) throws Exception {
    SubmarineRpcServer server = startRpcServer();
    server.blockUntilShutdown();
  }

  public static SubmarineRpcServer startRpcServer() throws IOException {
    SubmarineConfiguration submarineConfiguration =
        SubmarineConfiguration.getInstance();
    int rpcServerPort = submarineConfiguration.getInt(
        SubmarineConfiguration.ConfVars.SUBMARINE_SERVER_REMOTE_EXECUTION_PORT);
    SubmarineRpcServer server = new SubmarineRpcServer(rpcServerPort);
    server.start();
    return server;
  }

  /**
   * <p>See SubmarineServerProtocol.proto for details of the methods.
   */
  protected static class SubmarineServerRpcService
      extends SubmarineServerProtocolGrpc.SubmarineServerProtocolImplBase {

    @Override
    public void submitJob(ParameterProto request,
        io.grpc.stub.StreamObserver<ApplicationIdProto> responseObserver) {
      LOG.info("Start to submit a job.");
      RpcContext rpcContext =
          SubmarineRpcServerProto.convertParameterProtoToRpcContext(request);
      Parameter parameter =
          SubmarineRpcServerProto.convertParameterProtoToParameter(request);
      ClientContext clientContext = getClientContext(rpcContext);
      ApplicationId applicationId = null;
      try {
        applicationId = run(clientContext, parameter);
      } catch (IOException | YarnException | SubmarineException e) {
        LOG.error(e.getMessage(), e);
      }
      responseObserver.onNext(SubmarineRpcServerProto.
          convertApplicationIdToApplicationIdProto(applicationId));
      responseObserver.onCompleted();
    }

    @Override
    public void testRpc(ParametersHolderProto request,
        io.grpc.stub.StreamObserver<ApplicationIdProto> responseObserver) {
      responseObserver.onNext(checkFeature(request));
      responseObserver.onCompleted();
    }

    private ApplicationIdProto checkFeature(ParametersHolderProto request) {
      LOG.debug(request.toString());
      return ApplicationIdProto.newBuilder().setApplicationId("application_1_1").build();
    }

    protected ApplicationId run(ClientContext clientContext, Parameter parameter)
        throws IOException, YarnException, SubmarineException {
      // TODO replaced with JobManager
      JobSubmitter jobSubmitter =
          clientContext.getRuntimeFactory().getJobSubmitterInstance();
      ApplicationId applicationId = jobSubmitter.submitJob(parameter);
      return applicationId;
    }
  }

}
