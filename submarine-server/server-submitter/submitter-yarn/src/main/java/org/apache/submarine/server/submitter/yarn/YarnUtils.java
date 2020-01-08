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

package org.apache.submarine.server.submitter.yarn;

import com.linkedin.tony.Constants;
import com.linkedin.tony.TonyConfigurationKeys;
import com.linkedin.tony.util.Utils;
import org.apache.commons.cli.ParseException;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.yarn.api.records.Resource;
import org.apache.hadoop.yarn.exceptions.YarnException;
import org.apache.submarine.client.cli.CliConstants;
import org.apache.submarine.client.cli.param.Localization;
import org.apache.submarine.client.cli.param.ParametersHolder;
import org.apache.submarine.commons.runtime.param.Parameter;
import org.apache.submarine.commons.runtime.resource.ResourceUtils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import static org.apache.submarine.commons.utils.SubmarineConfiguration.SUBMARINE_RUNTIME_APP_TYPE;

/**
 * Utilities for YARN Runtime.
 */
public final class YarnUtils {
  private static final Log LOG = LogFactory.getLog(YarnUtils.class);

  public static Configuration tonyConfFromClientContext(
          ParametersHolder parameters) throws YarnException, ParseException {
    Configuration tonyConf = new Configuration();
    // Add tony.xml for configuration.
    tonyConf.addResource(Constants.TONY_XML);
    tonyConf.setStrings(TonyConfigurationKeys.FRAMEWORK_NAME,
            parameters.getFramework().getValue());
    tonyConf.setStrings(TonyConfigurationKeys.APPLICATION_NAME,
            parameters.getParameters().getName());
    tonyConf.setStrings(
        TonyConfigurationKeys.getInstancesKey(Constants.WORKER_JOB_NAME),
            parameters.getOptionValue(CliConstants.N_WORKERS));
    if (parameters.getOptionValue(CliConstants.N_PS) != null) {
      tonyConf.setStrings(
              TonyConfigurationKeys.getInstancesKey(Constants.PS_JOB_NAME),
              parameters.getOptionValue(CliConstants.N_PS));
    }
    // Resources for PS & Worker
    if (parameters.getOptionValue(CliConstants.PS_RES) != null) {
      Resource resource = getResource(parameters, CliConstants.PS_RES);

      tonyConf.setInt(
          TonyConfigurationKeys.getResourceKey(Constants.PS_JOB_NAME,
              Constants.VCORES),
              resource.getVirtualCores());
      tonyConf.setLong(
          TonyConfigurationKeys.getResourceKey(Constants.PS_JOB_NAME,
              Constants.MEMORY),
          ResourceUtils.getMemorySize(resource));
    }
    if (parameters.getOptionValue(CliConstants.WORKER_RES) != null) {
      Resource resource = getResource(parameters, CliConstants.WORKER_RES);

      tonyConf.setInt(
          TonyConfigurationKeys.getResourceKey(Constants.WORKER_JOB_NAME,
              Constants.VCORES),
              resource.getVirtualCores());
      tonyConf.setLong(
          TonyConfigurationKeys.getResourceKey(Constants.WORKER_JOB_NAME,
              Constants.MEMORY),
          ResourceUtils.getMemorySize(resource));
      tonyConf.setLong(
          TonyConfigurationKeys.getResourceKey(Constants.WORKER_JOB_NAME,
              Constants.GPUS),
          ResourceUtils.getResourceValue(resource,
              ResourceUtils.GPU_URI));
    }
    if (parameters.getOptionValue(CliConstants.QUEUE) != null) {
      tonyConf.set(
          TonyConfigurationKeys.YARN_QUEUE_NAME,
              parameters.getOptionValue(CliConstants.QUEUE));
    }
    // Set up Docker for PS & Worker
    if (parameters.getOptionValue(CliConstants.DOCKER_IMAGE) != null) {
      tonyConf.set(TonyConfigurationKeys.getContainerDockerKey(),
              parameters.getOptionValue(CliConstants.DOCKER_IMAGE));
      tonyConf.setBoolean(TonyConfigurationKeys.DOCKER_ENABLED, true);
    }
    if (parameters.getOptionValue(CliConstants.WORKER_DOCKER_IMAGE) != null) {
      tonyConf.set(
          TonyConfigurationKeys.getDockerImageKey(Constants.WORKER_JOB_NAME),
              parameters.getOptionValue(CliConstants.WORKER_DOCKER_IMAGE));
      tonyConf.setBoolean(TonyConfigurationKeys.DOCKER_ENABLED, true);
    }
    if (parameters.getOptionValue(CliConstants.PS_DOCKER_IMAGE) != null) {
      tonyConf.set(
          TonyConfigurationKeys.getDockerImageKey(Constants.PS_JOB_NAME),
              parameters.getOptionValue(CliConstants.PS_DOCKER_IMAGE));
      tonyConf.setBoolean(TonyConfigurationKeys.DOCKER_ENABLED, true);
    }

    // Set up container environment
    if (parameters.getOptionValues(CliConstants.ENV) != null) {
      List<String> envs = parameters.getOptionValues(CliConstants.ENV);
      tonyConf.setStrings(
              TonyConfigurationKeys.CONTAINER_LAUNCH_ENV,
              envs.toArray(new String[0]));
      tonyConf.setStrings(TonyConfigurationKeys.EXECUTION_ENV,
              envs.stream()
                      .map(env -> env.replaceAll("DOCKER_", ""))
                      .toArray(String[]::new));
      tonyConf.setStrings(TonyConfigurationKeys.CONTAINER_LAUNCH_ENV,
              envs.stream().map(env -> env.replaceAll("DOCKER_", ""))
                      .toArray(String[]::new));
    }
    tonyConf.setStrings(TonyConfigurationKeys.APPLICATION_TYPE, SUBMARINE_RUNTIME_APP_TYPE);
    // Set up running command
    if (parameters.getOptionValue(CliConstants.WORKER_LAUNCH_CMD) != null) {
      tonyConf.set(
          TonyConfigurationKeys.getExecuteCommandKey(Constants.WORKER_JOB_NAME),
              parameters.getOptionValue(CliConstants.WORKER_LAUNCH_CMD));
    }

    if (parameters.getOptionValue(CliConstants.PS_LAUNCH_CMD) != null) {
      tonyConf.set(
          TonyConfigurationKeys.getExecuteCommandKey(Constants.PS_JOB_NAME),
              parameters.getOptionValue(CliConstants.PS_LAUNCH_CMD));
    }

    tonyConf.setBoolean(TonyConfigurationKeys.SECURITY_ENABLED,
        !parameters.hasOption(CliConstants.INSECURE_CLUSTER));

    // Set up container resources
    if (parameters.getOptionValues(CliConstants.LOCALIZATION) != null) {
      List<String> localizationsStr = parameters
              .getOptionValues(CliConstants.LOCALIZATION);
      List<Localization> localizations = new ArrayList<>();
      for (String loc : localizationsStr) {
        Localization localization = new Localization();
        localization.parse(loc);
        localizations.add(localization);
      }

      tonyConf.setStrings(TonyConfigurationKeys.getContainerResourcesKey(),
              localizations.stream()
              .map(lo -> lo.getRemoteUri() + Constants.RESOURCE_DIVIDER
                  + lo.getLocalPath())
              .toArray(String[]::new));
    }

    if (parameters.getOptionValues(CliConstants.ARG_CONF) != null) {
      String[] confArray = parameters
              .getOptionValues(CliConstants.ARG_CONF).toArray(new String[0]);
      for (Map.Entry<String, String> cliConf : Utils
          .parseKeyValue(confArray).entrySet()) {
        String[] existingValue = tonyConf.getStrings(cliConf.getKey());
        if (existingValue != null
            && TonyConfigurationKeys
            .MULTI_VALUE_CONF.contains(cliConf.getKey())) {
          ArrayList<String> newValues = new ArrayList<>(Arrays
              .asList(existingValue));
          newValues.add(cliConf.getValue());
          tonyConf.setStrings(cliConf.getKey(),
                newValues.toArray(new String[0]));
        } else {
          tonyConf.set(cliConf.getKey(), cliConf.getValue());
        }
      }
    }

    LOG.info("Resources: " + tonyConf.get(
        TonyConfigurationKeys.getContainerResourcesKey()));
    return tonyConf;
  }

  private YarnUtils() {
  }

  private static Resource getResource(Parameter parametersHolder, String option)
          throws ParseException, YarnException {
    String ResourceStr =
            parametersHolder.getOptionValue(option);
    if (ResourceStr == null) {
      throw new ParseException("--" + option + " is absent.");
    }
    return ResourceUtils.createResourceFromString(ResourceStr);
  }
}
