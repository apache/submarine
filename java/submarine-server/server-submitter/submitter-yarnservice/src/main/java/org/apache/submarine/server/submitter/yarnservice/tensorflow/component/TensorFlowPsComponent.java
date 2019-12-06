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

package org.apache.submarine.server.submitter.yarnservice.tensorflow.component;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.yarn.service.api.records.Component;
import org.apache.submarine.client.cli.param.runjob.RunJobParameters;
import org.apache.submarine.client.cli.param.runjob.TensorFlowRunJobParameters;
import org.apache.submarine.commons.runtime.api.TensorFlowRole;
import org.apache.submarine.commons.runtime.fs.RemoteDirectoryManager;
import org.apache.submarine.server.submitter.yarnservice.AbstractComponent;
import org.apache.submarine.server.submitter.yarnservice.FileSystemOperations;
import org.apache.submarine.server.submitter.yarnservice.command.TensorFlowLaunchCommandFactory;
import org.apache.submarine.server.submitter.yarnservice.tensorflow.TensorFlowCommons;
import org.apache.submarine.server.submitter.yarnservice.utils.DockerUtilities;
import org.apache.submarine.server.submitter.yarnservice.utils.SubmarineResourceUtils;

import java.io.IOException;
import java.util.Objects;

/**
 * Component implementation for TensorFlow's PS process.
 */
public class TensorFlowPsComponent extends AbstractComponent {
  public TensorFlowPsComponent(FileSystemOperations fsOperations,
                               RemoteDirectoryManager remoteDirectoryManager,
                               TensorFlowLaunchCommandFactory launchCommandFactory,
                               RunJobParameters parameters,
                               Configuration yarnConfig) {
    super(fsOperations, remoteDirectoryManager, parameters,
        TensorFlowRole.PS, yarnConfig, launchCommandFactory);
  }

  @Override
  public Component createComponent() throws IOException {
    TensorFlowRunJobParameters tensorFlowParams =
        (TensorFlowRunJobParameters) this.parameters;

    Objects.requireNonNull(tensorFlowParams.getPsResource(),
        "PS resource must not be null!");
    if (tensorFlowParams.getNumPS() < 1) {
      throw new IllegalArgumentException("Number of PS should be at least 1!");
    }

    Component component = new Component();
    component.setName(role.getComponentName());
    component.setNumberOfContainers((long) tensorFlowParams.getNumPS());
    component.setRestartPolicy(Component.RestartPolicyEnum.NEVER);
    component.setResource(
        SubmarineResourceUtils.convertYarnResourceToServiceResource(tensorFlowParams.getPsResource()));

    // Override global docker image if needed.
    if (tensorFlowParams.getPsDockerImage() != null) {
      component.setArtifact(
          DockerUtilities.getDockerArtifact(tensorFlowParams.getPsDockerImage()));
    }
    TensorFlowCommons.addCommonEnvironments(component, role);
    generateLaunchCommand(component);

    return component;
  }
}
