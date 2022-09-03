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

package org.apache.submarine.server.submitter.k8s.model.seldon;

import org.apache.submarine.commons.utils.exception.SubmarineRuntimeException;
import org.apache.submarine.server.api.model.ServeSpec;

/**
 * SeldonDeployment K8s Model Resource Factory
 */
public class SeldonDeploymentFactory {

  /**
   * Get SeldonDeployment by model type
   */
  public static SeldonResource getSeldonDeployment(ServeSpec spec) throws SubmarineRuntimeException {
    String modelId = spec.getModelId();
    String resourceName = String.format("submarine-model-%s-%s", spec.getId(), modelId);

    String modelName = spec.getModelName();
    String modelType = spec.getModelType();
    String modelURI = spec.getModelURI();
    Integer modelVersion = spec.getModelVersion();

    switch (modelType) {
      case "tensorflow":
        return new SeldonDeploymentTFServing(resourceName, modelName, modelVersion,
            modelId, modelURI);
      case "pytorch":
        return new SeldonDeploymentPytorchServing(resourceName, modelName, modelVersion,
            modelId, modelURI);
      case "xgboost":// TODO(cdmikechen): Will fix https://issues.apache.org/jira/browse/SUBMARINE-1316
      default:
        throw new SubmarineRuntimeException("Given serve type: " + modelType + " is not supported.");
    }
  }
}
