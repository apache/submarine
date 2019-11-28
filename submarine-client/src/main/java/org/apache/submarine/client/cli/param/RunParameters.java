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

package org.apache.submarine.client.cli.param;

import com.google.common.collect.Lists;
import org.apache.commons.cli.ParseException;
import org.apache.hadoop.yarn.exceptions.YarnException;
import org.apache.submarine.client.cli.CliConstants;
import org.apache.submarine.commons.runtime.param.BaseParameters;
import org.apache.submarine.commons.runtime.ClientContext;
import org.apache.submarine.commons.runtime.param.Parameter;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Parameters required to run anything on cluster. Such as run job / serve model
 */
public abstract class RunParameters extends BaseParameters {
  private String savedModelPath;
  private String dockerImageName;
  private List<String> envVars = Lists.newArrayList();
  private String queue;

  @Override
  protected void updateParameters(Parameter parametersHolder,
      ClientContext clientContext)
      throws ParseException, IOException, YarnException {
    this.savedModelPath = parametersHolder.getOptionValue(
        CliConstants.SAVED_MODEL_PATH);
    this.envVars = getEnvVars(parametersHolder);
    this.queue = parametersHolder.getOptionValue(CliConstants.QUEUE);
    this.dockerImageName = parametersHolder.getOptionValue(
        CliConstants.DOCKER_IMAGE);

    super.updateParameters(parametersHolder, clientContext);
  }

  private List<String> getEnvVars(ParametersHolder parametersHolder)
      throws YarnException {
    List<String> result = Lists.newArrayList();
    List<String> envVarsArray = parametersHolder.getOptionValues(
        CliConstants.ENV);
    if (envVarsArray != null) {
      result.addAll(envVarsArray);
    }
    return result;
  }

  public String getQueue() {
    return queue;
  }

  public String getDockerImageName() {
    return dockerImageName;
  }

  public List<String> getEnvVars() {
    return envVars;
  }

  public String getSavedModelPath() {
    return savedModelPath;
  }

  public void setSavedModelPath(String savedModelPath) {
    this.savedModelPath = savedModelPath;
  }

  public RunParameters setEnvVars(List<String> envVars) {
    this.envVars = envVars;
    return this;
  }
}
