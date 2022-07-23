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

package org.apache.submarine.server.submitter.k8s.model.mljob;

import org.apache.submarine.server.api.exception.InvalidSpecException;
import org.apache.submarine.server.api.spec.ExperimentMeta;
import org.apache.submarine.server.api.spec.ExperimentSpec;
import org.apache.submarine.server.submitter.k8s.model.pytorchjob.PyTorchJob;
import org.apache.submarine.server.submitter.k8s.model.tfjob.TFJob;
import org.apache.submarine.server.submitter.k8s.model.xgboostjob.XGBoostJob;

/**
 * Select different MLJob implementation classes according to different framework
 */
public class MLJobFactory {

  /**
   * Get MLJob by framework name
   */
  public static MLJob getMLJob(ExperimentSpec experimentSpec) {
    String frameworkName = experimentSpec.getMeta().getFramework();
    ExperimentMeta.SupportedMLFramework framework = ExperimentMeta.SupportedMLFramework
            .valueOfName(frameworkName);
    switch (framework) {
      case TENSORFLOW:
        return new TFJob(experimentSpec);
      case PYTORCH:
        return new PyTorchJob(experimentSpec);
      case XGBOOST:
        return new XGBoostJob(experimentSpec);
      default:
        throw new InvalidSpecException("Unsupported framework name: " + frameworkName +
                ". Supported frameworks are: " +
                String.join(",", ExperimentMeta.SupportedMLFramework.names()));
    }
  }

}
