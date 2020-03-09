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

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.Option;
import org.apache.commons.lang3.StringUtils;
import org.apache.hadoop.yarn.api.records.ApplicationId;
import org.apache.hadoop.yarn.api.records.Resource;
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
import org.apache.submarine.commons.rpc.CommandLineProto;
import org.apache.submarine.commons.rpc.ListOfString;
import org.apache.submarine.commons.rpc.LocalizationProto;
import org.apache.submarine.commons.rpc.OptionProto;
import org.apache.submarine.commons.rpc.ParameterProto;
import org.apache.submarine.commons.rpc.PyTorchRunJobParameterProto;
import org.apache.submarine.commons.rpc.QuicklinkProto;
import org.apache.submarine.commons.rpc.ResourceProto;
import org.apache.submarine.commons.rpc.RoleParameterProto;
import org.apache.submarine.commons.rpc.RunParameterProto;
import org.apache.submarine.commons.rpc.ShowJobParameterProto;
import org.apache.submarine.commons.rpc.TensorFlowRunJobParameterProto;
import org.apache.submarine.commons.runtime.Framework;
import org.apache.submarine.commons.runtime.api.PyTorchRole;
import org.apache.submarine.commons.runtime.api.Role;
import org.apache.submarine.commons.runtime.api.TensorFlowRole;
import org.apache.submarine.commons.runtime.param.Parameter;
import org.apache.submarine.commons.runtime.resource.ResourceUtils;

import java.lang.reflect.Constructor;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class SubmarineRpcServerProto {

  public static Parameter convertParameterProtoToParameter(
      ParameterProto parameterProto) {
    Parameter parameter = null;
    if (parameterProto.hasPytorchRunJobParameter()) {
      parameter = convertParameterProtoToPyTorchRunJob(parameterProto);
    } else if (parameterProto.hasTensorflowRunJobParameter()) {
      parameter = convertParameterProtoToTensorFlowRunJob(parameterProto);
    } else if (parameterProto.hasShowJobParameter()) {
      parameter = convertParameterProtoToShowJob(parameterProto);
    }
    setCommandLineYamlConfigIfNeeded(parameter, parameterProto);
    return parameter;
  }

  public static void setCommandLineYamlConfigIfNeeded(
      Parameter parameter, ParameterProto parameterProto) {
    if (parameter instanceof ParametersHolder) {
      ParametersHolder parametersHolder = ((ParametersHolder) parameter);
      CommandLine commandLine = convertCommandLineProtoToCommandLine(
          parameterProto.getCommandLine());
      parametersHolder.setParsedCommandLine(commandLine);
      parametersHolder.setYamlStringConfigs(
          parameterProto.getYamlStringConfigsMap());
      parametersHolder.setYamlListConfigs(
          covertYamlListConfigs(parameterProto.getYamlListConfigsMap()));
    }
  }

  public static Map<String, List<String>> covertYamlListConfigs(
      Map<String, ListOfString> yamlListConfigs) {
    Map<String, List<String>> map = new HashMap<>();
    for (Map.Entry<String, ListOfString> entry : yamlListConfigs.entrySet()) {
      List<String> value =
          entry.getValue().getValuesList();
      map.put(entry.getKey(), value);
    }
    return map;
  }

  public static CommandLine convertCommandLineProtoToCommandLine(
      CommandLineProto commandLineProto) {
    CommandLine commandLine;
    Class<CommandLine> clz = CommandLine.class;
    try {
      Constructor<CommandLine> c = clz.getDeclaredConstructor();
      c.setAccessible(true);
      commandLine = c.newInstance();
    } catch (Exception e) {
      throw new RuntimeException(e.getMessage(), e);
    }

    for (OptionProto optionProto : commandLineProto.getOptionsList()) {
      Option option = new Option(optionProto.getOpt(), "");
      try {
        Class optionClass = Option.class;
        Method add = optionClass.getDeclaredMethod("add", String.class);
        add.setAccessible(true);
        for (String value : optionProto.getValuesList()) {
          add.invoke(option, value);
        }
      } catch (Exception e) {
        throw new RuntimeException(e.getMessage(), e.getCause());
      }
      try {
        Method getOption = clz.getDeclaredMethod("addOption", Option.class);
        getOption.setAccessible(true);
        getOption.invoke(commandLine, option);
      } catch (Exception e) {
        throw new RuntimeException(e.getMessage(), e);
      }
    }
    return commandLine;
  }

  public static Parameter convertParameterProtoToPyTorchRunJob(
      ParameterProto parameterProto) {
    Framework framework = Framework.parseByValue(parameterProto.getFramework());
    PyTorchRunJobParameterProto pyTorchRunJobParameterProto =
        parameterProto.getPytorchRunJobParameter();
    PyTorchRunJobParameters runJobParameters = new PyTorchRunJobParameters();

    Parameter parameter = convertRunParametersProtoToParameter(runJobParameters,
        pyTorchRunJobParameterProto.getRunParameterProto(), framework);
    return parameter;
  }

  public static Parameter convertParameterProtoToTensorFlowRunJob(
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

  public static Parameter convertParameterProtoToShowJob(
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

  public static RpcContext convertParameterProtoToRpcContext(
      ParameterProto parameterProto) {
    RpcContext rpcContext = new RpcContext();
    if (parameterProto.getSubmarineJobConfigMapMap() != null) {
      rpcContext.setSubmarineJobConfigMap(
          parameterProto.getSubmarineJobConfigMapMap());
    }
    return rpcContext;
  }

  public static Parameter convertRunParametersProtoToParameter(
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

  public static RoleParameters convertRoleParameterProtoToRoleParameters(
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

  public static List<Localization> convertLocalizationProtoToLocalization(
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

  public static List<Quicklink> convertQuicklinkProtoToQuicklink(
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

  public static Resource convertResourceProtoToResource(
      ResourceProto resourceProto) {
    Resource resource = ResourceUtils.createResource(
        resourceProto.getResourceMapMap());
    return resource;
  }

  public ParameterProto convertShowJobToParameterProto(Parameter parameters) {
    ShowJobParameterProto showJobproto = ShowJobParameterProto.newBuilder()
        .setName(parameters.getParameters().getName())
        .build();
    ParameterProto parameterProto = ParameterProto.newBuilder()
        .setShowJobParameter(showJobproto)
        .setFramework(parameters.getFramework().getValue())
        .build();
    return parameterProto;
  }

  public static ApplicationIdProto convertApplicationIdToApplicationIdProto(
      ApplicationId applicationId) {
    String application = applicationId != null ? applicationId.toString() : "";
    ApplicationIdProto applicationIdProto =
        ApplicationIdProto.newBuilder().setApplicationId(
            application).build();
    return applicationIdProto;
  }

}
