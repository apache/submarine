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

package org.apache.submarine.server.manager;

import com.google.common.annotations.VisibleForTesting;
import java.util.List;
import javax.ws.rs.core.Response;

import org.apache.submarine.commons.utils.exception.SubmarineRuntimeException;
import org.apache.submarine.server.database.model.entities.ModelVersionEntity;
import org.apache.submarine.server.database.model.entities.RegisteredModelEntity;
import org.apache.submarine.server.database.model.entities.RegisteredModelTagEntity;
import org.apache.submarine.server.database.model.service.ModelVersionService;
import org.apache.submarine.server.database.model.service.RegisteredModelService;

import org.apache.submarine.server.database.model.service.RegisteredModelTagService;
import org.apache.submarine.server.s3.Client;

/**
 * Registered model manager.
 */
public class RegisteredModelManager {
  private static RegisteredModelManager manager;
  /* Registered model service */
  private final RegisteredModelService registeredModelService;

  /* Model version service */
  private final ModelVersionService modelVersionService;

  /* Registered model tag service */
  private final RegisteredModelTagService registeredModelTagService;

  private final Client s3Client;

  /**
   * Get the singleton instance.
   *
   * @return object
   */
  public static synchronized RegisteredModelManager getInstance() {
    if (manager == null) {
      manager = new RegisteredModelManager(new RegisteredModelService(), new ModelVersionService(),
          new RegisteredModelTagService(), new Client());
    }
    return manager;
  }

  @VisibleForTesting
  protected RegisteredModelManager(RegisteredModelService registeredModelService,
      ModelVersionService modelVersionService, RegisteredModelTagService registeredModelTagService,
      Client s3Client) {
    this.registeredModelService = registeredModelService;
    this.modelVersionService = modelVersionService;
    this.registeredModelTagService = registeredModelTagService;
    this.s3Client = s3Client;
  }

  /**
   * Create registered model.
   *
   * @param entity spec
   */

  public void createRegisteredModel(RegisteredModelEntity entity) throws SubmarineRuntimeException {
    checkRegisteredModel(entity);
    registeredModelService.insert(entity);
  }

  /**
   * Get detailed info about the registered model by registered model name.
   *
   * @param name registered model name
   * @return detailed info about the registered model
   */
  public RegisteredModelEntity getRegisteredModel(String name) throws SubmarineRuntimeException {
    return registeredModelService.selectWithTag(name);
  }

  /**
   * List registered models.
   *
   * @return list
   */
  public List<RegisteredModelEntity> listRegisteredModels() throws SubmarineRuntimeException {
    return registeredModelService.selectAll();
  }

  /**
   * Update the registered model with registered model name.
   *
   * @param name   old registered model name
   * @param entity registered model entity
   */
  public void updateRegisteredModel(String name, RegisteredModelEntity entity)
      throws SubmarineRuntimeException {
    RegisteredModelEntity oldRegisteredModelEntity = registeredModelService.select(name);
    if (oldRegisteredModelEntity == null) {
      throw new SubmarineRuntimeException(Response.Status.NOT_FOUND.getStatusCode(),
        "Invalid. Registered model " + name + " is not existed.");
    }
    checkRegisteredModel(entity);
    if (!name.equals(entity.getName())) {
      registeredModelService.rename(name, entity.getName());
    }
    registeredModelService.update(entity);
  }

  /**
   * Delete the registered model with registered model name.
   *
   * @param name registered model name
   */
  public void deleteRegisteredModel(String name) throws SubmarineRuntimeException {
    List<ModelVersionEntity> modelVersions = modelVersionService.selectAllVersions(name);
    modelVersions.forEach(modelVersion -> {
      String stage = modelVersion.getCurrentStage();
      if (stage.equals("Production")) {
        throw new SubmarineRuntimeException(Response.Status.NOT_ACCEPTABLE.getStatusCode(),
            "Invalid. Some version of models are in the production stage");
      }
    });
    deleteModelInS3(modelVersions);
    registeredModelService.delete(name);
  }

  /**
   * Create a registered model tag.
   *
   * @param name registered model name
   * @param tag  tag name
   */
  public void createRegisteredModelTag(String name, String tag) throws SubmarineRuntimeException {
    checkRegisteredModelTag(name, tag);
    RegisteredModelTagEntity registeredModelTag = new RegisteredModelTagEntity();
    registeredModelTag.setName(name);
    registeredModelTag.setTag(tag);
    registeredModelTagService.insert(registeredModelTag);
  }

  /**
   * Delete a registered model tag.
   *
   * @param name registered model name
   * @param tag  tag name
   */
  public void deleteRegisteredModelTag(String name, String tag) throws SubmarineRuntimeException {
    checkRegisteredModelTag(name, tag);
    RegisteredModelTagEntity registeredModelTag = new RegisteredModelTagEntity();
    registeredModelTag.setName(name);
    registeredModelTag.setTag(tag);
    registeredModelTagService.delete(registeredModelTag);
  }

  /**
   * Check if registered model spec is valid spec.
   *
   * @param entity registered model entity
   */
  private void checkRegisteredModel(RegisteredModelEntity entity) {
    if (entity == null) {
      throw new SubmarineRuntimeException(Response.Status.OK.getStatusCode(),
          "Invalid. Registered model entity object is null.");
    }
    if (entity.getName() == null || entity.getName().equals("")) {
      throw new SubmarineRuntimeException(Response.Status.OK.getStatusCode(),
          "Invalid. Registered model name is null.");
    }
  }

  private void deleteModelInS3(List<ModelVersionEntity> modelVersions) throws SubmarineRuntimeException {
    try {
      modelVersions.forEach(modelVersion -> s3Client.deleteArtifactsByModelVersion(
          modelVersion.getName(),
          modelVersion.getVersion(),
          modelVersion.getId()
      )
      );
    } catch (SubmarineRuntimeException e) {
      throw new SubmarineRuntimeException(Response.Status.INTERNAL_SERVER_ERROR.getStatusCode(),
            "Some error happen when deleting the model in s3 bucket.");
    }
  }

  /**
   * Check if registered model tag is valid spec.
   *
   * @param name registered model name
   * @param tag  tag name
   */
  private void checkRegisteredModelTag(String name, String tag) {
    if (name.equals("")) {
      throw new SubmarineRuntimeException(Response.Status.OK.getStatusCode(),
          "Invalid. Registered model name is null.");
    }
    if (tag.equals("")) {
      throw new SubmarineRuntimeException(Response.Status.OK.getStatusCode(),
          "Invalid. Tag name is null.");
    }
    RegisteredModelEntity registeredModel = registeredModelService.select(name);
    if (registeredModel == null){
      throw new SubmarineRuntimeException(Response.Status.NOT_FOUND.getStatusCode(),
          "Invalid. Registered model " + name + " is not existed.");
    }
  }

}
