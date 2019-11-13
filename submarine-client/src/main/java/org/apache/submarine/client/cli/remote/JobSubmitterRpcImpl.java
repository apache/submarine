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

import com.google.common.annotations.VisibleForTesting;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import io.grpc.StatusRuntimeException;
import org.apache.hadoop.yarn.api.records.ApplicationId;
import org.apache.hadoop.yarn.api.records.Resource;
import org.apache.hadoop.yarn.exceptions.YarnException;
import org.apache.submarine.client.cli.param.Localization;
import org.apache.submarine.client.cli.param.Quicklink;
import org.apache.submarine.client.cli.param.RunParameters;
import org.apache.submarine.client.cli.param.ShowJobParameters;
import org.apache.submarine.client.cli.param.runjob.PyTorchRunJobParameters;
import org.apache.submarine.client.cli.param.runjob.RunJobParameters;
import org.apache.submarine.client.cli.param.runjob.TensorFlowRunJobParameters;
import org.apache.submarine.client.cli.runjob.RoleParameters;
import org.apache.submarine.commons.rpc.ApplicationIdProto;
import org.apache.submarine.commons.rpc.LocalizationProto;
import org.apache.submarine.commons.rpc.ParameterProto;
import org.apache.submarine.commons.rpc.PyTorchRunJobParameterProto;
import org.apache.submarine.commons.rpc.QuicklinkProto;
import org.apache.submarine.commons.rpc.ResourceProto;
import org.apache.submarine.commons.rpc.RoleParameterProto;
import org.apache.submarine.commons.rpc.RunParameterProto;
import org.apache.submarine.commons.rpc.ShowJobParameterProto;
import org.apache.submarine.commons.rpc.SubmarineServerProtocolGrpc;
import org.apache.submarine.commons.rpc.SubmarineServerProtocolGrpc.SubmarineServerProtocolBlockingStub;
import org.apache.submarine.commons.rpc.SubmarineServerProtocolGrpc.SubmarineServerProtocolStub;
import org.apache.submarine.commons.rpc.TensorFlowRunJobParameterProto;
import org.apache.submarine.commons.runtime.ClientContext;
import org.apache.submarine.commons.runtime.JobSubmitter;
import org.apache.submarine.commons.runtime.param.BaseParameters;
import org.apache.submarine.commons.runtime.param.Parameter;
import org.apache.submarine.commons.runtime.resource.ResourceUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;

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
    ParameterProto request = convertParameterToParameterProto(parameters);

    ApplicationId applicationId = null;
    try {
      ApplicationIdProto applicationIdProto = blockingStub.submitJob(request);
      applicationId =
          convertApplicationIdProtoToApplicationId(applicationIdProto);
    } catch (StatusRuntimeException e) {
      LOG.error(e.getMessage(),e);
    }
    return applicationId;
  }

  @VisibleForTesting
  ParameterProto convertParameterToParameterProto(Parameter parameters) {
    ParameterProto proto = null;
    BaseParameters baseParameters = parameters.getParameters();
    if (baseParameters instanceof RunParameters) {
      if (baseParameters instanceof PyTorchRunJobParameters) {
        // Handle pyTorch job parameters
        proto = convertPyTorchRunJobToParameterProto(parameters);
      } else if (baseParameters instanceof TensorFlowRunJobParameters) {
        // Handle tensorflow job parameters
        proto = convertTensorFlowRunJobToParameterProto(parameters);
      }
    } else if (baseParameters instanceof ShowJobParameters) {
      // Handle show job parameters
      proto = convertShowJobToParameterProto(parameters);
    }
    return proto;
  }

  private ParameterProto convertPyTorchRunJobToParameterProto(
      Parameter parameters) {
    PyTorchRunJobParameterProto pytorchProto =
        PyTorchRunJobParameterProto.newBuilder()
            .setRunParameterProto(
                convertParameterToRunParametersProto(parameters)).build();
    ParameterProto parameterProto = ParameterProto.newBuilder()
        .setPytorchRunJobParameter(pytorchProto)
        .setFramework(parameters.getFramework().getValue())
        .putAllSubmarineJobConfigMap(rpcContext.getSubmarineJobConfigMap())
        .build();
    return parameterProto;
  }

  private ParameterProto convertTensorFlowRunJobToParameterProto(
      Parameter parameters) {
    TensorFlowRunJobParameters tensorFlowRunJobParameters =
        (TensorFlowRunJobParameters)parameters.getParameters();
    TensorFlowRunJobParameterProto tfProto =
        TensorFlowRunJobParameterProto.newBuilder()
            .setRunParameterProto(
                convertParameterToRunParametersProto(parameters))
            .setTensorboardEnabled(
                tensorFlowRunJobParameters.isTensorboardEnabled())
            .setPsParameter(convertRoleParametersToRoleParameterProto(
                tensorFlowRunJobParameters.getPsParameters()))
            .setTensorBoardParameter(convertRoleParametersToRoleParameterProto(
                tensorFlowRunJobParameters.getTensorBoardParameters()))
            .build();
    ParameterProto parameterProto = ParameterProto.newBuilder()
        .setTensorflowRunJobParameter(tfProto)
        .setFramework(parameters.getFramework().getValue())
        .putAllSubmarineJobConfigMap(rpcContext.getSubmarineJobConfigMap())
        .build();
    return parameterProto;
  }

  private RunParameterProto convertParameterToRunParametersProto(
      Parameter parameters) {
    RunJobParameters runJobParameter =
        (RunJobParameters) parameters.getParameters();
    RunParameterProto runParametersProto = RunParameterProto.newBuilder()
        .setCheckpointPath(
            Optional.ofNullable(runJobParameter.getCheckpointPath()).orElse(""))
        .addAllConfPairs(runJobParameter.getConfPairs())
        .setDistributed(runJobParameter.isDistributed())
        .setDistributeKeytab(runJobParameter.isDistributeKeytab())
        .setDockerImageName(
            Optional.ofNullable(runJobParameter.getDockerImageName()).orElse(""))
        .addAllEnvars(runJobParameter.getEnvars())
        .setInput(Optional.ofNullable(runJobParameter.getInputPath()).orElse(""))
        .setKeytab(Optional.ofNullable(runJobParameter.getKeytab()).orElse(""))
        .addAllLocalizations(convertLocalizationToLocalizationProto(
            runJobParameter.getLocalizations()))
        .setName(Optional.ofNullable(runJobParameter.getName()).orElse(""))
        .setPrincipal(
            Optional.ofNullable(runJobParameter.getPrincipal()).orElse(""))
        .setQueue(
            Optional.ofNullable(runJobParameter.getQueue()).orElse(""))
        .addAllQuicklinks(convertQuicklinktoQuicklinkProto(
            runJobParameter.getQuicklinks()))
        .setSavedModelPath(
            Optional.ofNullable(runJobParameter.getSavedModelPath()).orElse(""))
        .setSecurityDisabled(runJobParameter.isSecurityDisabled())
        .setWaitJobFinish(runJobParameter.isWaitJobFinish())
        .setWorkerParameter(convertRoleParametersToRoleParameterProto(
            runJobParameter.getWorkerParameter()))
        .build();
    return runParametersProto;
  }

  private List<LocalizationProto> convertLocalizationToLocalizationProto(
      List<Localization> localizations) {
    List<LocalizationProto> LocalizationProtos =
        new ArrayList<LocalizationProto>();
    for (Localization localization: localizations) {
      LocalizationProtos.add(LocalizationProto.newBuilder()
          .setLocalPath(localization.getLocalPath())
          .setMountPermission(localization.getMountPermission())
          .setRemoteUri(localization.getRemoteUri())
          .build());
    }
    return LocalizationProtos;
  }

  private List<QuicklinkProto> convertQuicklinktoQuicklinkProto(
      List<Quicklink> quicklinks) {
    List<QuicklinkProto> quicklinkProtos =
        new ArrayList<QuicklinkProto>();
    for (Quicklink quicklink: quicklinks) {
      quicklinkProtos.add(QuicklinkProto.newBuilder()
          .setLabel(quicklink.getLabel())
          .setComponentInstanceName(quicklink.getComponentInstanceName())
          .setProtocol(quicklink.getProtocol())
          .setPort(quicklink.getPort())
          .build());
    }
    return quicklinkProtos;
  }

  private RoleParameterProto convertRoleParametersToRoleParameterProto(
      RoleParameters roleParameter) {
    RoleParameterProto roleParameterProto = RoleParameterProto.newBuilder()
        .setRole(roleParameter.getRole().getName())
        .setDockerImage(
            Optional.ofNullable(roleParameter.getDockerImage()).orElse(""))
        .setReplicas(roleParameter.getReplicas())
        .setLaunchCommand(
            Optional.ofNullable(roleParameter.getLaunchCommand()).orElse(""))
        .setResourceProto(
            convertResourceToResourceProto(roleParameter.getResource()))
        .build();
    return roleParameterProto;
  }

  private ResourceProto convertResourceToResourceProto(Resource resource) {
    Map<String, Long> map = ResourceUtils.getResourceMap(resource);
    return ResourceProto.newBuilder().putAllResourceMap(map).build();
  }

  private ParameterProto convertShowJobToParameterProto(Parameter parameters) {
    ShowJobParameterProto showJobproto = ShowJobParameterProto.newBuilder()
        .setName(parameters.getParameters().getName())
        .build();
    ParameterProto parameterProto = ParameterProto.newBuilder()
        .setShowJobParameter(showJobproto)
        .setFramework(parameters.getFramework().getValue())
        .putAllSubmarineJobConfigMap(rpcContext.getSubmarineJobConfigMap())
        .build();
    return parameterProto;
  }

  private ApplicationId convertApplicationIdProtoToApplicationId(
      ApplicationIdProto applicationIdProto) {
    return fromString(applicationIdProto.getApplicationId());
  }

  /**
   * As hadoop-2.7 doesn't have this method, we add this method in submarine.
   * @param appIdStr
   * @return
   */
  private ApplicationId fromString(String appIdStr) {
    String APPLICATION_ID_PREFIX = "application_";
    if (!appIdStr.startsWith(APPLICATION_ID_PREFIX)) {
      throw new IllegalArgumentException("Invalid ApplicationId prefix: "
          + appIdStr + ". The valid ApplicationId should start with prefix "
          + "application");
    }
    try {
      int pos1 = APPLICATION_ID_PREFIX.length() - 1;
      int pos2 = appIdStr.indexOf('_', pos1 + 1);
      if (pos2 < 0) {
        throw new IllegalArgumentException("Invalid ApplicationId: "
            + appIdStr);
      }
      long rmId = Long.parseLong(appIdStr.substring(pos1 + 1, pos2));
      int appId = Integer.parseInt(appIdStr.substring(pos2 + 1));
      ApplicationId applicationId = ApplicationId.newInstance(rmId, appId);
      return applicationId;
    } catch (NumberFormatException n) {
      throw new IllegalArgumentException("Invalid ApplicationId: "
          + appIdStr, n);
    }
  }

}
