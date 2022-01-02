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

package org.apache.submarine.server.model;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import javax.ws.rs.core.Response;

import org.apache.submarine.commons.utils.exception.SubmarineRuntimeException;
import org.apache.submarine.server.SubmitterManager;
import org.apache.submarine.server.api.Submitter;
import org.apache.submarine.server.api.model.ServeResponse;
import org.apache.submarine.server.api.model.ServeSpec;
import org.apache.submarine.server.model.database.entities.ModelVersionEntity;
import org.apache.submarine.server.model.database.service.ModelVersionService;



public class ModelManager {
  private static final Logger LOG = LoggerFactory.getLogger(ModelManager.class);

  private static ModelManager manager;

  private final Submitter submitter;

  private final ModelVersionService modelVersionService;

  private ModelManager(Submitter submitter, ModelVersionService modelVersionService) {
    this.submitter = submitter;
    this.modelVersionService = modelVersionService;
  }

  /**
   * Get the singleton instance.
   *
   * @return object
   */
  public static ModelManager getInstance() {
    if (manager == null) {
      synchronized (ModelManager.class) {
        manager = new ModelManager(SubmitterManager.loadSubmitter(), new ModelVersionService());
      }
    }
    return manager;
  }

  /**
   * Create a model serve.
   */
  public ServeResponse createServe(ServeSpec spec) throws SubmarineRuntimeException {
    setServeInfo(spec);


    LOG.info("Create {} + model serve", spec.getModelType());

    submitter.createServe(spec);

    return getServeResponse();
  }

  /**
   * Delete a model serve.
   */
  public void deleteServe(ServeSpec spec) throws SubmarineRuntimeException {
    setServeInfo(spec);

    LOG.info("Delete {} model serve", spec.getModelType());

    submitter.deleteServe(spec);
  }

  private void checkServeSpec(ServeSpec spec) throws SubmarineRuntimeException {
    if (spec == null) {
      throw new SubmarineRuntimeException(Response.Status.OK.getStatusCode(),
              "Invalid. Serve Spec object is null.");
    } else {
      if (spec.getModelName() == null) {
        throw new SubmarineRuntimeException(Response.Status.OK.getStatusCode(),
                "Invalid. Model name in Serve Soec is null.");
      }
      Integer modelVersion = spec.getModelVersion();
      if (modelVersion == null || modelVersion <= 0) {
        throw new SubmarineRuntimeException(Response.Status.OK.getStatusCode(),
                "Invalid. Model version must be positive, but get " + modelVersion);
      }
    }
  }

  private void setServeInfo(ServeSpec spec){
    checkServeSpec(spec);

    // Get model type and model uri from DB and set the value in the spec.
    ModelVersionEntity modelVersion = modelVersionService.select(spec.getModelName(), spec.getModelVersion());
    spec.setModelURI(modelVersion.getSource());
    spec.setModelType(modelVersion.getModelType());
  }

  private ServeResponse getServeResponse(){
    return new ServeResponse();
  }
}
