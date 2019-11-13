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

import io.grpc.Server;
import io.grpc.ServerBuilder;
import org.apache.commons.lang3.StringUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.yarn.api.records.ApplicationId;
import org.apache.hadoop.yarn.api.records.Resource;
import org.apache.hadoop.yarn.conf.YarnConfiguration;
import org.apache.hadoop.yarn.exceptions.YarnException;
import org.apache.submarine.client.cli.param.Localization;
import org.apache.submarine.client.cli.param.ParametersHolder;
import org.apache.submarine.client.cli.param.Quicklink;
import org.apache.submarine.client.cli.param.ShowJobParameters;
import org.apache.submarine.client.cli.param.runjob.PyTorchRunJobParameters;
import org.apache.submarine.client.cli.param.runjob.RunJobParameters;
import org.apache.submarine.client.cli.param.runjob.TensorFlowRunJobParameters;
import org.apache.submarine.client.cli.remote.RpcContext;
import org.apache.submarine.client.cli.runjob.RoleParameters;
import org.apache.submarine.commons.rpc.ApplicationIdProto;
import org.apache.submarine.commons.rpc.LocalizationProto;
import org.apache.submarine.commons.rpc.ParameterProto;
import org.apache.submarine.commons.rpc.ParametersHolderProto;
import org.apache.submarine.commons.rpc.PyTorchRunJobParameterProto;
import org.apache.submarine.commons.rpc.QuicklinkProto;
import org.apache.submarine.commons.rpc.ResourceProto;
import org.apache.submarine.commons.rpc.RoleParameterProto;
import org.apache.submarine.commons.rpc.RunParameterProto;
import org.apache.submarine.commons.rpc.ShowJobParameterProto;
import org.apache.submarine.commons.rpc.SubmarineServerProtocolGrpc;
import org.apache.submarine.commons.rpc.TensorFlowRunJobParameterProto;
import org.apache.submarine.commons.runtime.ClientContext;
import org.apache.submarine.commons.runtime.Framework;
import org.apache.submarine.commons.runtime.JobSubmitter;
import org.apache.submarine.commons.runtime.RuntimeFactory;
import org.apache.submarine.commons.runtime.api.PyTorchRole;
import org.apache.submarine.commons.runtime.api.Role;
import org.apache.submarine.commons.runtime.api.TensorFlowRole;
import org.apache.submarine.commons.runtime.param.Parameter;
import org.apache.submarine.commons.runtime.resource.ResourceUtils;
import org.apache.submarine.commons.utils.SubmarineConfiguration;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * A sample gRPC server that serve the RouteGuide (see route_guide.proto) service.
 */
public class SubmarineRpcServer {
  private static final Logger LOG = LoggerFactory.getLogger(
      SubmarineRpcServer.class.getName());

  private final int port;
  private final Server server;

  public SubmarineRpcServer(int port) throws IOException {
    this(ServerBuilder.forPort(port), port);
  }

  /** Create a RouteGuide server using serverBuilder as a base and features as data. */
  public SubmarineRpcServer(ServerBuilder<?> serverBuilder, int port) {
    this.port = port;
    server = serverBuilder.addService(new SubmarineServerRpcService())
        .build();
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

  public static SubmarineRpcServer startRpcServer()
      throws IOException, InterruptedException {
    SubmarineConfiguration submarineConfiguration =
        SubmarineConfiguration.getInstance();
    int rpcServerPort = submarineConfiguration.getInt(
        SubmarineConfiguration.ConfVars.SUBMARINE_SERVER_REMOTE_EXECUTION_PORT);
    SubmarineRpcServer server = new SubmarineRpcServer(rpcServerPort);
    server.start();
    return server;
  }

  /**
   * Our implementation of RouteGuide service.
   *
   * <p>See route_guide.proto for details of the methods.
   */
  private static class SubmarineServerRpcService
      extends SubmarineServerProtocolGrpc.SubmarineServerProtocolImplBase {

    @Override
    public void submitJob(ParameterProto request,
        io.grpc.stub.StreamObserver<ApplicationIdProto> responseObserver) {
      LOG.info("Start to submit a job.");
      RpcContext rpcContext = convertParameterProtoToRpcContext(request);
      Parameter parameter = convertParameterProtoToParameter(request);
      ClientContext clientContext = getClientContext(rpcContext);
      ApplicationId applicationId = null;
      try {
        applicationId = run(clientContext, parameter);
      } catch (IOException | YarnException e) {
        LOG.error(e.getMessage(), e);
      }
      responseObserver.onNext(convertApplicationIdToApplicationIdProto(
          applicationId));
      responseObserver.onCompleted();
    }

    @Override
    public void testRpc(ParametersHolderProto request,
        io.grpc.stub.StreamObserver<ApplicationIdProto> responseObserver) {
      responseObserver.onNext(checkFeature(request));
      responseObserver.onCompleted();
    }

    private ApplicationIdProto checkFeature(ParametersHolderProto request) {
      LOG.info(request.toString());
      return ApplicationIdProto.newBuilder().setApplicationId("application_1").build();
    }

    private ApplicationId run(ClientContext clientContext, Parameter parameter)
        throws IOException, YarnException {
      JobSubmitter jobSubmitter =
          clientContext.getRuntimeFactory().getJobSubmitterInstance();
      ApplicationId applicationId = jobSubmitter.submitJob(parameter);
      return applicationId;
    }
  }

  private static Parameter convertParameterProtoToParameter(
      ParameterProto parameterProto) {
    Parameter parameter = null;
    if (parameterProto.hasPytorchRunJobParameter()) {
      parameter = convertParameterProtoToPyTorchRunJob(parameterProto);
    } else if (parameterProto.hasTensorflowRunJobParameter()) {
      parameter = convertParameterProtoToTensorFlowRunJob(parameterProto);
    } else if (parameterProto.hasShowJobParameter()) {
      parameter = convertParameterProtoToShowJob(parameterProto);
    }
    return parameter;
  }

  private static Parameter convertParameterProtoToPyTorchRunJob(
      ParameterProto parameterProto) {
    Framework framework = Framework.parseByValue(parameterProto.getFramework());
    PyTorchRunJobParameterProto pyTorchRunJobParameterProto =
        parameterProto.getPytorchRunJobParameter();
    PyTorchRunJobParameters runJobParameters = new PyTorchRunJobParameters();

    Parameter parameter = convertRunParametersProtoToParameter(runJobParameters,
        pyTorchRunJobParameterProto.getRunParameterProto(), framework);
    return parameter;
  }

  private static Parameter convertParameterProtoToTensorFlowRunJob(
      ParameterProto parameterProto) {
    TensorFlowRunJobParameterProto tensorFlowRunJobParameterProto =
        parameterProto.getTensorflowRunJobParameter();
    Framework framework = Framework.parseByValue(parameterProto.getFramework());

    TensorFlowRunJobParameters runJobParameters =
        new TensorFlowRunJobParameters();
    runJobParameters.setTensorboardEnabled(
        tensorFlowRunJobParameterProto.getTensorboardEnabled());
    runJobParameters.setPsParameters(convertRoleParameterProtoToRoleParameters(
        tensorFlowRunJobParameterProto.getPsParameter(), framework));
    runJobParameters.setTensorBoardParameters(
        convertRoleParameterProtoToRoleParameters(
        tensorFlowRunJobParameterProto.getTensorBoardParameter(), framework));

    Parameter parameter = convertRunParametersProtoToParameter(runJobParameters,
        tensorFlowRunJobParameterProto.getRunParameterProto(), framework);
    parameter.setFramework(framework);
    return parameter;
  }

  private static Parameter convertParameterProtoToShowJob(
      ParameterProto parameterProto) {
    Framework framework = Framework.parseByValue(parameterProto.getFramework());
    ShowJobParameterProto showJobParameterProto =
        parameterProto.getShowJobParameter();
    ShowJobParameters showJobParameters = new ShowJobParameters();
    showJobParameters.setName(showJobParameterProto.getName());

    ParametersHolder parameter = ParametersHolder.create();
    parameter.setParameters(showJobParameters);
    parameter.setFramework(framework);
    return parameter;
  }

  private static RpcContext convertParameterProtoToRpcContext(
      ParameterProto parameterProto) {
    RpcContext rpcContext = new RpcContext();
    if (parameterProto.getSubmarineJobConfigMapMap() != null) {
      rpcContext.setSubmarineJobConfigMap(
          parameterProto.getSubmarineJobConfigMapMap());
    }
    return rpcContext;
  }

  private static Parameter convertRunParametersProtoToParameter(
      RunJobParameters runJobParameters, RunParameterProto runJobParameterProto,
      Framework framework) {
    if (StringUtils.isNotBlank(runJobParameterProto.getCheckpointPath())) {
      runJobParameters.setCheckpointPath(
          runJobParameterProto.getCheckpointPath());
    }
    if (StringUtils.isNotBlank(runJobParameterProto.getDockerImageName())) {
      runJobParameters.setDockerImageName(
          runJobParameterProto.getDockerImageName());
    }
    if (StringUtils.isNotBlank(runJobParameterProto.getInput())) {
      runJobParameters.setInputPath(runJobParameterProto.getInput());
    }
    if (StringUtils.isNotBlank(runJobParameterProto.getKeytab())) {
      runJobParameters.setKeytab(runJobParameterProto.getKeytab());
    }
    if (StringUtils.isNotBlank(runJobParameterProto.getName())) {
      runJobParameters.setName(runJobParameterProto.getName());
    }
    if (StringUtils.isNotBlank(runJobParameterProto.getPrincipal())) {
      runJobParameters.setPrincipal(runJobParameterProto.getPrincipal());
    }
    if (StringUtils.isNotBlank(runJobParameterProto.getQueue())) {
      runJobParameters.setQueue(runJobParameterProto.getQueue());
    }
    if (StringUtils.isNotBlank(runJobParameterProto.getSavedModelPath())) {
      runJobParameters.setSavedModelPath(
          runJobParameterProto.getSavedModelPath());
    }

    runJobParameters.setConfPairs(runJobParameterProto.getConfPairsList())
        .setDistributed(runJobParameterProto.getDistributed())
        .setDistributeKeytab(runJobParameterProto.getDistributeKeytab())
        .setLocalizations(convertLocalizationProtoToLocalization(
            runJobParameterProto.getLocalizationsList()))
        .setQuicklinks(convertQuicklinkProtoToQuicklink(
            runJobParameterProto.getQuicklinksList()))
        .setSecurityDisabled(runJobParameterProto.getSecurityDisabled())
        .setWaitJobFinish(runJobParameterProto.getWaitJobFinish())
        .setWorkerParameter(convertRoleParameterProtoToRoleParameters(
            runJobParameterProto.getWorkerParameter(), framework));
    runJobParameters.setEnvars(runJobParameterProto.getEnvarsList());

    ParametersHolder parameter = ParametersHolder.create();
    parameter.setParameters(runJobParameters);
    parameter.setFramework(framework);
    return parameter;
  }

  private static RoleParameters convertRoleParameterProtoToRoleParameters(
      RoleParameterProto roleParameterProto, Framework framework) {
    Role role = null;
    switch (framework) {
      case TENSORFLOW:
        role = TensorFlowRole.valueOf(roleParameterProto.getRole());
        break;
      case PYTORCH:
        role = PyTorchRole.valueOf(roleParameterProto.getRole());
        break;
    }

    RoleParameters roleParameters = RoleParameters.createEmpty(role);
    if (StringUtils.isNotBlank(roleParameterProto.getDockerImage())) {
      roleParameters.setDockerImage(roleParameterProto.getDockerImage());
    }
    if (StringUtils.isNotBlank(roleParameterProto.getLaunchCommand())) {
      roleParameters.setLaunchCommand(roleParameterProto.getLaunchCommand());
    }

    roleParameters.setReplicas(roleParameterProto.getReplicas())
        .setResource(convertResourceProtoToResource(
            roleParameterProto.getResourceProto()));
    return roleParameters;
  }

  private static List<Localization> convertLocalizationProtoToLocalization(
      List<LocalizationProto> localizationsList) {
    List<Localization> localizations = new ArrayList<Localization>();
    for (LocalizationProto localizationProto: localizationsList) {
      Localization localization = new Localization();
      localization.setLocalPath(localizationProto.getLocalPath())
          .setMountPermission(localizationProto.getMountPermission())
          .setRemoteUri(localizationProto.getRemoteUri());
      localizations.add(localization);
    }
    return localizations;
  }

  private static List<Quicklink> convertQuicklinkProtoToQuicklink(
      List<QuicklinkProto> quicklinksList) {
    List<Quicklink> quicklinks = new ArrayList<Quicklink>();
    for (QuicklinkProto quicklinkProto: quicklinksList) {
      Quicklink quicklink = new Quicklink();
      quicklink.setLabel(quicklinkProto.getLabel())
          .setComponentInstanceName(quicklinkProto.getComponentInstanceName())
          .setProtocol(quicklinkProto.getProtocol())
          .setPort(quicklinkProto.getPort());
      quicklinks.add(quicklink);
    }
    return quicklinks;
  }

  private static Resource convertResourceProtoToResource(
      ResourceProto resourceProto) {
    Resource resource = ResourceUtils.createResource(
        resourceProto.getResourceMapMap());
    return resource;
  }

  private ParameterProto convertShowJobToParameterProto(Parameter parameters) {
    ShowJobParameterProto showJobproto = ShowJobParameterProto.newBuilder()
        .setName(parameters.getParameters().getName())
        .build();
    ParameterProto parameterProto = ParameterProto.newBuilder()
        .setShowJobParameter(showJobproto)
        .setFramework(parameters.getFramework().getValue())
        .build();
    return parameterProto;
  }

  private static ApplicationIdProto convertApplicationIdToApplicationIdProto(
      ApplicationId applicationId) {
    String application = applicationId != null ? applicationId.toString() : "";
    ApplicationIdProto applicationIdProto =
        ApplicationIdProto.newBuilder().setApplicationId(
            application).build();
    return applicationIdProto;
  }

}
