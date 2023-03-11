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

package org.apache.submarine.server.k8s.agent.reconciler;

import io.javaoperatorsdk.operator.api.reconciler.Context;
import io.javaoperatorsdk.operator.api.reconciler.ControllerConfiguration;
import io.javaoperatorsdk.operator.api.reconciler.Reconciler;
import io.javaoperatorsdk.operator.api.reconciler.UpdateControl;
import org.apache.submarine.server.api.common.CustomResourceType;
import org.apache.submarine.server.k8s.agent.model.training.resource.PyTorchJob;

/**
 * PyTorch Job Reconciler
 * <p>
 * Submarine will add `submarine-experiment-name` label when creating the experiment,
 * so we need to do the filtering.
 * <p>
 * Label selectors reference:
 * https://kubernetes.io/docs/concepts/overview/working-with-objects/labels/#api
 */
@ControllerConfiguration(
    labelSelector = "submarine-experiment-name",
    generationAwareEventProcessing = false
)
public class PyTorchJobReconciler extends JobReconciler<PyTorchJob> implements Reconciler<PyTorchJob> {

  @Override
  public UpdateControl<PyTorchJob> reconcile(PyTorchJob pyTorchJob, Context<PyTorchJob> context) {
    triggerStatus(pyTorchJob);
    return UpdateControl.noUpdate();
  }

  @Override
  public CustomResourceType type() {
    return CustomResourceType.PyTorchJob;
  }
}
