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

package org.apache.submarine.server.submitter.yarnservice.tensorflow;

import org.apache.submarine.client.cli.param.runjob.TensorFlowRunJobParameters;
import org.apache.submarine.commons.runtime.Framework;
import org.apache.submarine.commons.runtime.ClientContext;
import org.apache.submarine.server.submitter.yarnservice.AbstractServiceSpec;
import org.apache.submarine.server.submitter.yarnservice.ServiceWrapper;
import org.apache.submarine.server.submitter.yarnservice.tensorflow.component.TensorBoardComponent;
import org.apache.submarine.server.submitter.yarnservice.tensorflow.component.TensorFlowPsComponent;
import org.apache.submarine.server.submitter.yarnservice.FileSystemOperations;
import org.apache.submarine.server.submitter.yarnservice.command.TensorFlowLaunchCommandFactory;
import org.apache.submarine.server.submitter.yarnservice.utils.Localizer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

/**
 * This class contains all the logic to create an instance
 * of a Service object for TensorFlow.
 * Worker,PS and Tensorboard components are added to the Service
 * based on the value of the received RunJobParameters.
 */
public class TensorFlowServiceSpec extends AbstractServiceSpec {
  private static final Logger LOG =
      LoggerFactory.getLogger(TensorFlowServiceSpec.class);
  private final TensorFlowRunJobParameters tensorFlowParameters;

  public TensorFlowServiceSpec(TensorFlowRunJobParameters parameters,
      ClientContext clientContext, FileSystemOperations fsOperations,
      TensorFlowLaunchCommandFactory launchCommandFactory,
      Localizer localizer) {
    super(parameters, clientContext, fsOperations, launchCommandFactory,
        localizer);
    this.tensorFlowParameters = parameters;
  }

  @Override
  public ServiceWrapper create() throws IOException {
    LOG.info("Creating TensorFlow service spec");
    ServiceWrapper serviceWrapper = createServiceSpecWrapper();

    if (tensorFlowParameters.getNumWorkers() > 0) {
      addWorkerComponents(serviceWrapper, Framework.TENSORFLOW);
    }

    if (tensorFlowParameters.getNumPS() > 0) {
      addPsComponent(serviceWrapper);
    }

    if (tensorFlowParameters.isTensorboardEnabled()) {
      createTensorBoardComponent(serviceWrapper);
    }

    // After all components added, handle quicklinks
    handleQuicklinks(serviceWrapper.getService());

    return serviceWrapper;
  }

  private void createTensorBoardComponent(ServiceWrapper serviceWrapper)
      throws IOException {
    TensorBoardComponent tbComponent = new TensorBoardComponent(fsOperations,
        remoteDirectoryManager, parameters,
        (TensorFlowLaunchCommandFactory) launchCommandFactory, yarnConfig);
    serviceWrapper.addComponent(tbComponent);

    addQuicklink(serviceWrapper.getService(), TensorBoardComponent.TENSORBOARD_QUICKLINK_LABEL,
        tbComponent.getTensorboardLink());
  }

  private void addPsComponent(ServiceWrapper serviceWrapper)
      throws IOException {
    serviceWrapper.addComponent(
        new TensorFlowPsComponent(fsOperations, remoteDirectoryManager,
            (TensorFlowLaunchCommandFactory) launchCommandFactory,
            parameters, yarnConfig));
  }

}
