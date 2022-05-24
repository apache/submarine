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

package org.apache.submarine.server.rest;

import java.util.List;
import javax.ws.rs.Consumes;
import javax.ws.rs.DELETE;
import javax.ws.rs.DefaultValue;
import javax.ws.rs.GET;
import javax.ws.rs.PATCH;
import javax.ws.rs.POST;
import javax.ws.rs.Path;
import javax.ws.rs.PathParam;
import javax.ws.rs.Produces;
import javax.ws.rs.QueryParam;
import javax.ws.rs.core.MediaType;
import javax.ws.rs.core.Response;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.responses.ApiResponse;

import org.apache.submarine.commons.utils.exception.SubmarineRuntimeException;
import org.apache.submarine.server.database.model.entities.ModelVersionEntity;
import org.apache.submarine.server.manager.ModelVersionManager;
import org.apache.submarine.server.utils.response.JsonResponse;

/**
 * Model version REST API v1.
 */
@Path(RestConstants.V1 + "/" + RestConstants.MODEL_VERSION)
@Produces({ MediaType.APPLICATION_JSON + "; " + RestConstants.CHARSET_UTF8 })
public class ModelVersionRestApi {

  /* ModelVersion manager  */
  private final ModelVersionManager modelVersionManager = ModelVersionManager.getInstance();

  /**
   * Return the Pong message for test the connectivity.
   *
   * @return Pong message
   */
  @GET
  @Path(RestConstants.PING)
  @Consumes(MediaType.APPLICATION_JSON)
  @Operation(summary = "Ping submarine server", tags = {
      "model-version"}, description = "Return the Pong message for test the connectivity", responses = {
      @ApiResponse(responseCode = "200", description = "successful operation",
          content = @Content(schema = @Schema(implementation = String.class)))})
  public Response ping() {
    return new JsonResponse.Builder<String>(Response.Status.OK).success(true).result("Pong").build();
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
   * @return success message
   */
  @POST
  @Consumes({ RestConstants.MEDIA_TYPE_YAML, MediaType.APPLICATION_JSON })
  @Operation(summary = "Create a model version instance", tags = { "model-version" }, responses = {
      @ApiResponse(description = "successful operation",
                   content = @Content(schema = @Schema(implementation = JsonResponse.class)))})
  public Response createModelVersion(ModelVersionEntity entity,
                                     @QueryParam("baseDir") String baseDir) {
    try {
      modelVersionManager.createModelVersion(entity, baseDir);
      return new JsonResponse.Builder<String>(Response.Status.OK).success(true)
        .message("Create a model version instance").build();
    } catch (SubmarineRuntimeException e) {
      return parseModelVersionServiceException(e);
    }
  }

  /**
   * List all model versions under same registered model name.
   *
   * @param name registered model name
   * @return model version list
   */
  @GET
  @Path("/{name}")
  @Operation(summary = "List model versions", tags = {"model-version"}, responses = {
      @ApiResponse(description = "successful operation",
          content = @Content(schema = @Schema(implementation = JsonResponse.class)))})
  public Response listModelVersions(@PathParam(RestConstants.MODEL_VERSION_NAME) String name) {
    try {
      List<ModelVersionEntity> modelVersionList = modelVersionManager.listModelVersions(name);
      return new JsonResponse.Builder<List<ModelVersionEntity>>(Response.Status.OK).success(true)
          .message("List all model version instances").result(modelVersionList).build();
    } catch (SubmarineRuntimeException e) {
      return parseModelVersionServiceException(e);
    }
  }

  /**
   * Get detailed info about the model version by name and version.
   *
   * @param name    model version's name
   * @param version model version's version
   * @return detailed info about the model version
   */
  @GET
  @Path("/{name}/{version}")
  @Operation(summary = "Get detailed info about the model version", tags = {"model-version"}, responses = {
      @ApiResponse(description = "successful operation",
          content = @Content(schema = @Schema(implementation = JsonResponse.class))),
      @ApiResponse(responseCode = "404", description = "ModelVersionEntity not found")})
  public Response getModelVersion(@PathParam(RestConstants.MODEL_VERSION_NAME) String name,
                                  @PathParam(RestConstants.MODEL_VERSION_VERSION) Integer version) {
    try {
      ModelVersionEntity modelVersion = modelVersionManager.getModelVersion(name, version);
      return new JsonResponse.Builder<ModelVersionEntity>(Response.Status.OK).success(true)
          .message("Get the model version instance").result(modelVersion).build();
    } catch (SubmarineRuntimeException e) {
      return parseModelVersionServiceException(e);
    }
  }

  /**
   * Delete the model version with model version name and version.
   *
   * @param name    model version's name
   * @param version model version's version
   * @return success message
   */
  @DELETE
  @Path("/{name}/{version}")
  @Operation(summary = "Delete the model version", tags = {"model-version"}, responses = {
      @ApiResponse(description = "successful operation",
          content = @Content(schema = @Schema(implementation = JsonResponse.class))),
      @ApiResponse(responseCode = "404", description = "ModelVersionEntity not found")})
  public Response deleteModelVersion(@PathParam(RestConstants.MODEL_VERSION_NAME) String name,
                                     @PathParam(RestConstants.MODEL_VERSION_VERSION) Integer version) {
    try {
      modelVersionManager.deleteModelVersion(name, version);
      return new JsonResponse.Builder<String>(Response.Status.OK).success(true)
          .message("Delete the model version instance").build();
    } catch (SubmarineRuntimeException e) {
      return parseModelVersionServiceException(e);
    }
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
   * @return success message
   */
  @PATCH
  @Path("")
  @Operation(summary = "Update the model version", tags = {"model-version"}, responses = {
      @ApiResponse(description = "successful operation",
          content = @Content(schema = @Schema(implementation = JsonResponse.class))),
      @ApiResponse(responseCode = "404", description = "ModelVersionEntity not found")})
  public Response updateModelVersion(ModelVersionEntity entity) {
    try {
      modelVersionManager.updateModelVersion(entity);
      return new JsonResponse.Builder<String>(Response.Status.OK).success(true)
          .message("Update the model version instance").build();
    } catch (SubmarineRuntimeException e) {
      return parseModelVersionServiceException(e);
    }
  }

  /**
   * Create a model version tag.
   *
   * @param name    model version's name
   * @param version model version's version
   * @param tag     tag name
   * @return success message
   */
  @POST
  @Path("/tag")
  @Consumes({ RestConstants.MEDIA_TYPE_YAML, MediaType.APPLICATION_JSON })
  @Operation(summary = "Create a model version tag instance", tags = { "model-version" }, responses = {
      @ApiResponse(description = "successful operation",
          content = @Content(schema = @Schema(implementation = JsonResponse.class))) })
  public Response createModelVersionTag(@DefaultValue("") @QueryParam("name") String name,
                                        @DefaultValue("") @QueryParam("version") String version,
                                        @DefaultValue("") @QueryParam("tag") String tag) {
    try {
      modelVersionManager.createModelVersionTag(name, version, tag);
      return new JsonResponse.Builder<String>(Response.Status.OK).success(true)
          .message("Create a model version tag instance").build();
    } catch (SubmarineRuntimeException e) {
      return parseModelVersionServiceException(e);
    }
  }

  /**
   * Delete a model version tag.
   *
   * @param name    model version's name
   * @param version model version's version
   * @param tag     tag name
   * @return success message
   */
  @DELETE
  @Path("/tag")
  @Consumes({ RestConstants.MEDIA_TYPE_YAML, MediaType.APPLICATION_JSON })
  @Operation(summary = "Delete a model version tag instance", tags = { "model-version" }, responses = {
      @ApiResponse(description = "successful operation",
          content = @Content(schema = @Schema(implementation = JsonResponse.class))) })
  public Response deleteModelVersionTag(@DefaultValue("") @QueryParam("name") String name,
                                        @DefaultValue("") @QueryParam("version") String version,
                                        @DefaultValue("") @QueryParam("tag") String tag) {
    try {
      modelVersionManager.deleteModelVersionTag(name, version, tag);
      return new JsonResponse.Builder<String>(Response.Status.OK).success(true)
          .message("Delete a model version tag instance").build();
    } catch (SubmarineRuntimeException e) {
      return parseModelVersionServiceException(e);
    }
  }

  private Response parseModelVersionServiceException(SubmarineRuntimeException e) {
    return new JsonResponse.Builder<String>(e.getCode()).message(e.getMessage()).build();
  }
}
