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
import javax.ws.rs.GET;
import javax.ws.rs.Path;
import javax.ws.rs.PathParam;
import javax.ws.rs.Produces;
import javax.ws.rs.core.MediaType;
import javax.ws.rs.core.Response;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.responses.ApiResponse;


import org.apache.submarine.commons.utils.exception.SubmarineRuntimeException;
import org.apache.submarine.server.model.database.entities.ModelVersionEntity;
import org.apache.submarine.server.model.database.service.ModelVersionService;


import org.apache.submarine.server.response.JsonResponse;

/**
 * Model version REST API v1.
 */
@Path(RestConstants.V1 + "/" + RestConstants.MODEL_VERSION)
@Produces({ MediaType.APPLICATION_JSON + "; " + RestConstants.CHARSET_UTF8 })
public class ModelVersionRestApi {

  /* Model version service */
  private final ModelVersionService modelVersionService = new ModelVersionService();

  /**
   * Return the Pong message for test the connectivity.
   *
   * @return Pong message
   */
  @GET
  @Path(RestConstants.PING)
  @Consumes(MediaType.APPLICATION_JSON)
  @Operation(summary = "Ping submarine server", tags = {
      "model-version" }, description = "Return the Pong message for test the connectivity", responses = {
      @ApiResponse(responseCode = "200", description = "successful operation",
          content = @Content(schema = @Schema(implementation = String.class))) })
  public Response ping() {
    return new JsonResponse.Builder<String>(Response.Status.OK).success(true).result("Pong").build();
  }

  /**
   * List all model versions under same registered model.
   *
   * @param name model version name
   * @return model-version list
   */
  @GET
  @Path("/{name}")
  @Operation(summary = "List model versions", tags = { "model-version" }, responses = {
      @ApiResponse(description = "successful operation",
          content = @Content(schema = @Schema(implementation = JsonResponse.class))) })
  public Response listModelVersions(@PathParam(RestConstants.MODEL_VERSION_NAME) String name) {
    try {
      List<ModelVersionEntity> modelVersionList = modelVersionService.selectAllVersions(name);
      return new JsonResponse.Builder<List<ModelVersionEntity>>(Response.Status.OK).success(true)
          .message("List all model version instances").result(modelVersionList).build();
    } catch (SubmarineRuntimeException e) {
      return parseModelVersionServiceException(e);
    }
  }

  /**
   * Get detailed info about the model version by name and version.
   *
   * @param name    model version name
   * @param version model version version
   * @return detailed info about the model-version
   */
  @GET
  @Path("/{name}/{version}")
  @Operation(summary = "Get detailed info about the model version", tags = { "model-version" }, responses = {
      @ApiResponse(description = "successful operation",
          content = @Content(schema = @Schema(implementation = JsonResponse.class))),
      @ApiResponse(responseCode = "404", description = "ModelVersionEntity not found") })
  public Response getModelVersion(@PathParam(RestConstants.MODEL_VERSION_NAME) String name,
      @PathParam(RestConstants.MODEL_VERSION_VERSION) Integer version) {
    try {
      ModelVersionEntity modelVersion = modelVersionService.selectWithTag(name, version);
      return new JsonResponse.Builder<ModelVersionEntity>(Response.Status.OK).success(true)
          .message("Get the model version instance").result(modelVersion).build();
    } catch (SubmarineRuntimeException e) {
      return parseModelVersionServiceException(e);
    }
  }

  /**
   * Delete the model version with model version name and version.
   *
   * @param name    model version name
   * @param version model version version
   * @return seccess message
   */
  @DELETE
  @Path("/{name}/{version}")
  @Operation(summary = "Delete the model version", tags = { "model-version" }, responses = {
      @ApiResponse(description = "successful operation",
          content = @Content(schema = @Schema(implementation = JsonResponse.class))),
      @ApiResponse(responseCode = "404", description = "ModelVersionEntity not found") })
  public Response deleteModelVersion(@PathParam(RestConstants.MODEL_VERSION_NAME) String name,
      @PathParam(RestConstants.MODEL_VERSION_VERSION) Integer version) {
    try {
      modelVersionService.delete(name, version);
      return new JsonResponse.Builder<String>(Response.Status.OK).success(true)
          .message("Delete the model version instance").build();
    } catch (SubmarineRuntimeException e) {
      return parseModelVersionServiceException(e);
    }
  }

  private Response parseModelVersionServiceException(SubmarineRuntimeException e) {
    return new JsonResponse.Builder<String>(e.getCode()).message(e.getMessage()).build();
  }

}
