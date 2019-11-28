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
import org.apache.commons.cli.ParseException;
import org.apache.commons.lang3.StringUtils;
import org.apache.hadoop.yarn.api.records.Resource;
import org.apache.hadoop.yarn.exceptions.YarnException;
import org.apache.submarine.client.cli.CliConstants;
import org.apache.submarine.client.cli.RoleResourceParser;
import org.apache.submarine.client.cli.param.ParametersHolder;

import java.io.IOException;
import java.util.Objects;

/**
 * This class holds parameters for the TensorBoard role.
 */
public class TensorBoardParameters {
  private RoleResourceParser roleResourceParser;
  private ParametersHolder parametersHolder;
  private boolean enabled;
  private String dockerImage;
  private Resource resource;

  TensorBoardParameters(RoleResourceParser roleResourceParser,
                        ParametersHolder parametersHolder)
      throws YarnException, IOException, ParseException {
    Objects.requireNonNull(roleResourceParser, "RoleResourceParser " +
        "must not be null!");
    Objects.requireNonNull(parametersHolder, "ParametersHolder " +
        "must not be null!");
    this.roleResourceParser = roleResourceParser;
    this.parametersHolder = parametersHolder;
    this.enabled = false;

    if (this.parametersHolder.hasOption(CliConstants.TENSORBOARD)) {
      this.enabled = true;
      this.dockerImage = this.parametersHolder.getOptionValue(
          CliConstants.TENSORBOARD_DOCKER_IMAGE);
    }
    resource = parseTensorboardResource();
  }

  @VisibleForTesting
  TensorBoardParameters() {
  }

  private Resource parseTensorboardResource()
      throws YarnException, ParseException, IOException {
    return roleResourceParser.parseResource(CliConstants.TENSORBOARD_RESOURCES,
        getTensorBoardResourceStr());
  }

  private String getTensorBoardResourceStr() throws YarnException {
    String resourceStr = parametersHolder.getOptionValue(
        CliConstants.TENSORBOARD_RESOURCES);
    if (StringUtils.isEmpty(resourceStr) || !enabled) {
      resourceStr = CliConstants.TENSORBOARD_DEFAULT_RESOURCES;
    }
    return resourceStr;
  }

  boolean isEnabled() {
    return enabled;
  }

  public String getDockerImage() {
    return dockerImage;
  }

  public void setDockerImage(String dockerImage) {
    this.dockerImage = dockerImage;
  }

  public Resource getResource() {
    return resource;
  }

  public void setResource(Resource resource) {
    this.resource = resource;
  }
}