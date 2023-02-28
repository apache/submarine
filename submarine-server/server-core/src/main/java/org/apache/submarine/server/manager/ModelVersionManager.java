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
import org.json.JSONObject;
import java.util.List;
import javax.ws.rs.core.Response;

import org.apache.submarine.commons.utils.exception.SubmarineRuntimeException;
import org.apache.submarine.server.database.model.entities.ModelVersionEntity;
import org.apache.submarine.server.database.model.entities.ModelVersionTagEntity;
import org.apache.submarine.server.database.model.service.ModelVersionService;
import org.apache.submarine.server.database.model.service.ModelVersionTagService;
import org.apache.submarine.server.s3.Client;

/**
 * ModelVersion manager.
 */
public class ModelVersionManager {
  private static ModelVersionManager manager;

  /* Model version service */
  private final ModelVersionService modelVersionService;

  /* Model version tag service */
  private final ModelVersionTagService modelVersionTagService;

  private final Client s3Client;

  /**
   * Get the singleton instance.
   *
   * @return object
   */
  private static class ModelVersionManagerHolder {
    private static ModelVersionManager manager = new ModelVersionManager(ModelVersionService.getInstance(),
                                                                        new ModelVersionTagService(),
                                                                        Client.getInstance());
  }

  public static ModelVersionManager getInstance() {
    return ModelVersionManager.ModelVersionManagerHolder.manager;
  }

  @VisibleForTesting
  protected ModelVersionManager(ModelVersionService modelVersionService,
                                ModelVersionTagService modelVersionTagService, Client s3Client) {
    this.modelVersionService = modelVersionService;
    this.modelVersionTagService = modelVersionTagService;
    this.s3Client = s3Client;
  }

  /**
   * Create a model version.
   *
   * @param entity registered model entity
   * example: {
   *   "name": "example_name"
   *   "experimentId" : "4d4d02f06f6f437fa29e1ee8a9276d87"
   *   "userId": ""
   *   "description" : "example_description"
   *   "tags": ["123", "456"]
   * }
   * @param baseDir artifact base directory
   * example: "experiment/experiment-1643015349312-0001/1"
   */
  public void createModelVersion(ModelVersionEntity entity, String baseDir) throws SubmarineRuntimeException {
    String res = new String(s3Client.downloadArtifact(
        String.format("%s/description.json", baseDir)));
    JSONObject description = new JSONObject(res);
    String modelType =  description.get("model_type").toString();
    String id = description.get("id").toString();
    entity.setId(id);
    entity.setModelType(modelType);

    int version = modelVersionService.selectAllVersions(entity.getName()).stream().mapToInt(
        ModelVersionEntity::getVersion
    ).max().orElse(0) + 1;

    entity.setVersion(version);
    modelVersionService.insert(entity);

    // the directory of storing a single model must be unique for serving
    String uniqueModelPath = String.format("%s-%d-%s", entity.getName(), version, id);

    // copy artifacts
    s3Client.listAllObjects(baseDir).forEach(s -> {
      String relativePath = s.substring(String.format("%s/", baseDir).length());
      s3Client.copyArtifact(String.format("registry/%s/%s/%d/%s", uniqueModelPath,
          entity.getName(), entity.getVersion(), relativePath), s);
    });
  }

  /**
   * Get detailed info about the model version by name and version.
   *
   * @param name    model version's name
   * @param version model version's version
   * @return detailed info about the model version
   */
  public ModelVersionEntity getModelVersion(String name, Integer version) throws SubmarineRuntimeException {
    return modelVersionService.selectWithTag(name, version);
  }

  /**
   * List all model versions under same registered model name.
   *
   * @param name registered model name
   * @return model version list
   */
  public List<ModelVersionEntity> listModelVersions(String name) throws SubmarineRuntimeException {
    return modelVersionService.selectAllVersions(name);
  }

  /**
   * Update the model version.
   *
   * @param entity model version entity
   * example: {
   *   'name': 'example_name',
   *   'version': 1,
   *   'description': 'new_description',
   *   'currentStage': 'production',
   *   'dataset': 'new_dataset'
   * }
   */
  public void updateModelVersion(ModelVersionEntity entity) throws SubmarineRuntimeException {
    checkModelVersion(entity);
    modelVersionService.update(entity);
  }

  /**
   * Delete the model version with model version name and version.
   *
   * @param name    model version's name
   * @param version model version's version
   */
  public void deleteModelVersion(String name, Integer version) throws SubmarineRuntimeException {
    ModelVersionEntity spec = modelVersionService.select(name, version);
    s3Client.deleteArtifactsByModelVersion(name, version, spec.getId());
    modelVersionService.delete(name, version);
  }


  /**
   * Create a model version tag.
   *
   * @param name    model version's name
   * @param version model version's version
   * @param tag     tag name
   */
  public void createModelVersionTag(String name, String version, String tag)
      throws SubmarineRuntimeException {
    checkModelVersionTag(name, version, tag);
    ModelVersionTagEntity modelVersionTag = new ModelVersionTagEntity();
    modelVersionTag.setName(name);
    modelVersionTag.setVersion(Integer.parseInt(version));
    modelVersionTag.setTag(tag);
    modelVersionTagService.insert(modelVersionTag);
  }

  /**
   * Delete a model version tag.
   *
   * @param name    model version's name
   * @param version model version's version
   * @param tag     tag name
   */
  public void deleteModelVersionTag(String name, String version, String tag)
      throws SubmarineRuntimeException {
    checkModelVersionTag(name, version, tag);
    ModelVersionTagEntity modelVersionTag = new ModelVersionTagEntity();
    modelVersionTag.setName(name);
    modelVersionTag.setVersion(Integer.parseInt(version));
    modelVersionTag.setTag(tag);
    modelVersionTagService.delete(modelVersionTag);
  }

  /**
   * Check if model version tag is valid.
   *
   * @param name    model version's name
   * @param version model version's version
   * @param tag     tag name
   */
  private void checkModelVersionTag(String name, String version, String tag) {
    if (name.equals("")){
      throw new SubmarineRuntimeException(Response.Status.OK.getStatusCode(),
          "Invalid. Model version's name is null.");
    }
    if (version.equals("")) {
      throw new SubmarineRuntimeException(Response.Status.OK.getStatusCode(),
          "Invalid. Model version's version is null.");
    }
    int versionNum;
    try {
      versionNum = Integer.parseInt(version);
      if (versionNum < 1){
        throw new SubmarineRuntimeException(Response.Status.OK.getStatusCode(),
            "Invalid. Model version's version must be bigger than 0.");
      }
    } catch (NumberFormatException e){
      throw new SubmarineRuntimeException(Response.Status.OK.getStatusCode(),
          "Invalid. Model version's version must be an integer.");
    }
    if (tag.equals("")) {
      throw new SubmarineRuntimeException(Response.Status.OK.getStatusCode(),
          "Invalid. Tag name is null.");
    }
    ModelVersionEntity modelVersion = modelVersionService.select(name,
        versionNum);
    if (modelVersion == null){
      throw new SubmarineRuntimeException(Response.Status.NOT_FOUND.getStatusCode(),
          "Invalid. Model version " + name + " version " + versionNum + " is not existed.");
    }
  }

  private void checkModelVersion(ModelVersionEntity entity) {
    if (entity == null) {
      throw new SubmarineRuntimeException(Response.Status.OK.getStatusCode(),
          "Invalid. Model version entity object is null.");
    }
    if (entity.getName() == null || entity.getName().equals("")) {
      throw new SubmarineRuntimeException(Response.Status.OK.getStatusCode(),
          "Invalid. Model version's name is null.");
    }
    if (entity.getVersion() == null) {
      throw new SubmarineRuntimeException(Response.Status.OK.getStatusCode(),
          "Invalid. Model version's version is null.");
    }
    ModelVersionEntity modelVersion = modelVersionService.select(entity.getName(), entity.getVersion());
    if (modelVersion == null) {
      throw new SubmarineRuntimeException(Response.Status.NOT_FOUND.getStatusCode(),
          "Invalid. Model version entity with same name and version is not existed.");
    }
  }
}
