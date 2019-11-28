/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *     http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.submarine.client.cli.param.runjob;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableMap;
import org.apache.commons.cli.ParseException;
import org.apache.hadoop.yarn.api.records.Resource;
import org.apache.hadoop.yarn.exceptions.YarnException;
import org.apache.submarine.client.cli.CliConstants;
import org.apache.submarine.client.cli.RoleResourceParser;
import org.apache.submarine.client.cli.param.ParametersHolder;
import org.apache.submarine.commons.runtime.api.PyTorchRole;
import org.apache.submarine.commons.runtime.api.Role;
import org.apache.submarine.commons.runtime.api.TensorFlowRole;

import java.io.IOException;
import java.util.Map;
import java.util.Objects;

/**
 * Parses role related data from parameters.
 * Currently, we have Worker and PS (ParameterServer) role types.
 * Fields include number of instances the role has,
 * Resource object the role requires to operate with,
 * docker image and launch command of the role.
 * This class encapsulates data related to a particular Role.
 * Some examples: TF Worker process, TF PS process or a PyTorch worker process.
 */
public class RoleParameters {
  private final Role role;
  private final RoleResourceParser resourceParser;

  private static Map<Role, Integer> defaultInstanceCounts =
      ImmutableMap.of(
          TensorFlowRole.WORKER, 1,
          PyTorchRole.WORKER, 1,
          TensorFlowRole.PS, 0);

  private int replicas;
  private Resource resource;
  private String dockerImage;
  private String launchCommand;

  RoleParameters(Role role, RoleResourceParser resourceParser,
                 ParametersHolder parametersHolder)
      throws YarnException, IOException, ParseException {
    Objects.requireNonNull(role, "Role must not be null!");
    Objects.requireNonNull(resourceParser, "RoleResourceParser " +
        "must not be null!");
    Objects.requireNonNull(parametersHolder, "ParametersHolder " +
        "must not be null!");
    this.role = role;
    this.resourceParser = resourceParser;
    parse(parametersHolder);
  }

  @VisibleForTesting
  RoleParameters(Role role, RoleResourceParser resourceParser) {
    this.role = role;
    this.resourceParser = resourceParser;
  }

  private void parse(ParametersHolder paramHolder)
      throws YarnException, IOException, ParseException {
    this.replicas = getNumberOfInstances(paramHolder);
    this.resource = parseResource(paramHolder, replicas);
    this.dockerImage = getDockerImage(paramHolder);
    this.launchCommand = getLaunchCommand(paramHolder);
  }

  private int getNumberOfInstances(ParametersHolder parametersHolder)
      throws YarnException {
    int instanceCount = defaultInstanceCounts.get(role);
    String key = getNumberOfInstancesKey();

    if (parametersHolder.getOptionValue(key) != null) {
      instanceCount = Integer.parseInt(parametersHolder.getOptionValue(key));
    }
    return instanceCount;
  }

  private Resource parseResource(ParametersHolder parametersHolder,
                                 int instances) throws YarnException, ParseException, IOException {
    String key = getResourcesKey();
    String resourceStr = parametersHolder.getOptionValue(key);
    return resourceParser.parseResource(instances, key, resourceStr);
  }

  private String getDockerImage(ParametersHolder paramHolder)
      throws YarnException {
    return paramHolder.getOptionValue(getDockerImageKey());
  }

  private String getLaunchCommand(ParametersHolder paramHolder)
      throws YarnException {
    return paramHolder.getOptionValue(getLaunchCommandKey());
  }

  private String getLaunchCommandKey() {
    return role == TensorFlowRole.WORKER || role == PyTorchRole.WORKER
        ? CliConstants.WORKER_LAUNCH_CMD
        : CliConstants.PS_LAUNCH_CMD;
  }

  private String getNumberOfInstancesKey() {
    return role == TensorFlowRole.WORKER || role == PyTorchRole.WORKER
        ? CliConstants.N_WORKERS
        : CliConstants.N_PS;
  }

  private String getResourcesKey() {
    return role == TensorFlowRole.WORKER || role == PyTorchRole.WORKER
        ? CliConstants.WORKER_RES
        : CliConstants.PS_RES;
  }

  private String getDockerImageKey() {
    return role == TensorFlowRole.WORKER || role == PyTorchRole.WORKER
        ? CliConstants.WORKER_DOCKER_IMAGE
        : CliConstants.PS_DOCKER_IMAGE;
  }

  public Role getRole() {
    return role;
  }

  public int getReplicas() {
    return replicas;
  }

  public void setReplicas(int replicas) {
    this.replicas = replicas;
  }

  public Resource getResource() {
    return resource;
  }

  public void setResource(Resource resource) {
    this.resource = resource;
  }

  public String getDockerImage() {
    return dockerImage;
  }

  public void setDockerImage(String dockerImage) {
    this.dockerImage = dockerImage;
  }

  public String getLaunchCommand() {
    return launchCommand;
  }

  public void setLaunchCommand(String launchCommand) {
    this.launchCommand = launchCommand;
  }
}
