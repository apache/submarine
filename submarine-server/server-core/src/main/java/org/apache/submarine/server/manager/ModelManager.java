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

import org.json.JSONArray;
import org.json.JSONObject;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import javax.ws.rs.core.Response;

import org.apache.submarine.commons.utils.exception.SubmarineRuntimeException;
import org.apache.submarine.server.SubmitterManager;
import org.apache.submarine.server.api.Submitter;
import org.apache.submarine.server.api.model.ServeResponse;
import org.apache.submarine.server.api.model.ServeSpec;
import org.apache.submarine.server.api.proto.TritonModelConfig;
import org.apache.submarine.server.database.model.entities.ModelVersionEntity;
import org.apache.submarine.server.database.model.service.ModelVersionService;
import org.apache.submarine.server.s3.Client;
import org.apache.submarine.server.s3.S3Constants;


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
  private static class ModelManagerHolder {
    private static ModelManager manager = new ModelManager(SubmitterManager.loadSubmitter(),
                                                            ModelVersionService.getInstance());
  }

  public static ModelManager getInstance() {
    return ModelManager.ModelManagerHolder.manager;
  }

  /**
   * Create a model serve.
   */
  public ServeResponse createServe(ServeSpec spec) throws SubmarineRuntimeException {
    // Get model type and model uri from DB and set the value in the spec.
    ModelVersionEntity modelVersion = modelVersionService.select(spec.getModelName(), spec.getModelVersion());

    setServeInfo(spec, modelVersion);

    LOG.info("Create {} model serve.", spec.getModelType());

    if (spec.getModelType().equals("pytorch")){
      transferDescription(spec);
    }

    submitter.createServe(spec);

    modelVersion.setCurrentStage("Production");
    modelVersionService.update(modelVersion);

    return getServeResponse(spec);
  }

  /**
   * Delete a model serve.
   */
  public void deleteServe(ServeSpec spec) throws SubmarineRuntimeException {
    // Get model type and model uri from DB and set the value in the spec.
    ModelVersionEntity modelVersion = modelVersionService.select(spec.getModelName(), spec.getModelVersion());
    setServeInfo(spec, modelVersion);

    LOG.info("Delete {} model serve", spec.getModelType());

    submitter.deleteServe(spec);
    modelVersion.setCurrentStage("None");
    modelVersionService.update(modelVersion);
  }

  private void checkServeSpec(ServeSpec spec) throws SubmarineRuntimeException {
    if (spec == null) {
      throw new SubmarineRuntimeException(Response.Status.OK.getStatusCode(),
              "Invalid. Serve Spec object is null.");
    } else {
      if (spec.getModelName() == null) {
        throw new SubmarineRuntimeException(Response.Status.OK.getStatusCode(),
                "Invalid. Model name in Serve Spec is null.");
      }
      Integer modelVersion = spec.getModelVersion();
      if (modelVersion == null || modelVersion <= 0) {
        throw new SubmarineRuntimeException(Response.Status.OK.getStatusCode(),
                "Invalid. Model version must be positive, but get " + modelVersion);
      }
    }
  }

  private void setServeInfo(ServeSpec spec, ModelVersionEntity modelVersion){
    checkServeSpec(spec);

    String modelType = modelVersion.getModelType();
    String modelId = modelVersion.getId();
    spec.setModelType(modelType);
    spec.setModelId(modelId);

    String modelUniquePath = String.format("%s-%d-%s", spec.getModelName(), spec.getModelVersion(), modelId);
    if (spec.getModelType().equals("pytorch")) {
      spec.setModelURI(String.format("s3://%s/registry/%s", S3Constants.BUCKET, modelUniquePath));
    } else if (spec.getModelType().equals("tensorflow")) {
      spec.setModelURI(String.format("s3://%s/registry/%s/%s", S3Constants.BUCKET, modelUniquePath,
          spec.getModelName()));
    } else {
      throw new SubmarineRuntimeException(Response.Status.INTERNAL_SERVER_ERROR.getStatusCode(),
          String.format("Unexpected model type: %s", modelType));
    }
  }

  private void transferDescription(ServeSpec spec) {
    Client s3Client = Client.getInstance();
    String modelUniquePath = String.format("%s-%d-%s",
        spec.getModelName(), spec.getModelVersion(), spec.getModelId());
    String res  = new String(s3Client.downloadArtifact(String.format("registry/%s/%s/%d/description.json",
        modelUniquePath, spec.getModelName(), spec.getModelVersion())));
    JSONObject description = new JSONObject(res);

    TritonModelConfig.ModelConfig.Builder modelConfig = TritonModelConfig.ModelConfig.newBuilder();
    modelConfig.setPlatform("pytorch_libtorch");

    JSONArray inputs = (JSONArray) description.get("input");
    for (int idx = 0; idx < inputs.length(); idx++) {
      JSONArray dims = (JSONArray) ((JSONObject) inputs.get(idx)).get("dims");
      TritonModelConfig.ModelInput.Builder modelInput = TritonModelConfig.ModelInput.newBuilder();
      modelInput.setName("INPUT__" + idx);
      modelInput.setDataType(TritonModelConfig.DataType.valueOf("TYPE_FP32"));
      dims.forEach(dim -> modelInput.addDims((Integer) dim));
      modelConfig.addInput(modelInput);
    }

    JSONArray outputs = (JSONArray) description.get("output");
    for (int idx = 0; idx < outputs.length(); idx++) {
      JSONArray dims = (JSONArray) ((JSONObject) outputs.get(idx)).get("dims");
      TritonModelConfig.ModelOutput.Builder modelOutput = TritonModelConfig.ModelOutput.newBuilder();
      modelOutput.setName("OUTPUT__" + idx);
      modelOutput.setDataType(TritonModelConfig.DataType.valueOf("TYPE_FP32"));
      dims.forEach(dim -> modelOutput.addDims((Integer) dim));
      modelConfig.addOutput(modelOutput);
    }

    s3Client.logArtifact(String.format("registry/%s/%s/config.pbtxt", modelUniquePath, spec.getModelName()),
            modelConfig.toString().getBytes());
  }

  private ServeResponse getServeResponse(ServeSpec spec){
    ServeResponse serveResponse = new ServeResponse();
    if (spec.getModelType().equals("pytorch")) {
      serveResponse.setUrl(String.format("http://{submarine ip}/%s/%d/v2/models/%s/infer",
          spec.getModelName(), spec.getModelVersion(), spec.getModelName()));
    } else {
      serveResponse.setUrl(String.format("http://{submarine ip}/%s/%d/api/v1.0/predictions",
          spec.getModelName(), spec.getModelVersion()));
    }

    return serveResponse;
  }
}
