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
import javax.ws.rs.POST;
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
import org.apache.submarine.server.api.spec.RegisteredModelSpec;
import org.apache.submarine.server.model.database.entities.RegisteredModelEntity;
import org.apache.submarine.server.model.database.service.RegisteredModelService;

import org.apache.submarine.server.response.JsonResponse;




/**
 * Registered model REST API v1.
 */
@Path(RestConstants.V1 + "/" + RestConstants.REGISTERED_MODEL)
@Produces({ MediaType.APPLICATION_JSON + "; " + RestConstants.CHARSET_UTF8 })
public class RegisteredModelRestApi {

  /* Registered model service */
  private final RegisteredModelService registeredModelService = new RegisteredModelService();

  /**
   * Return the Pong message for test the connectivity.
   *
   * @return Pong message
   */
  @GET
  @Path(RestConstants.PING)
  @Consumes(MediaType.APPLICATION_JSON)
  @Operation(summary = "Ping submarine server", tags = {
      "registered-model" }, description = "Return the Pong message for test the connectivity", responses = {
      @ApiResponse(responseCode = "200", description = "successful operation",
          content = @Content(schema = @Schema(implementation = String.class))) })
  public Response ping() {
    return new JsonResponse.Builder<String>(Response.Status.OK).success(true).result("Pong").build();
  }

  /**
   * Create a registered model.
   *
   * @param spec registered model spec
   * @return success message
   */
  @POST
  @Consumes({ RestConstants.MEDIA_TYPE_YAML, MediaType.APPLICATION_JSON })
  @Operation(summary = "Create a registered model instance", tags = { "registered-model" }, responses = {
      @ApiResponse(description = "successful operation",
          content = @Content(schema = @Schema(implementation = JsonResponse.class))) })
  public Response createRegisteredModel(RegisteredModelSpec spec) {
    try {
      checkRegisteredModelSpec(spec);
      RegisteredModelEntity registeredModel = new RegisteredModelEntity();
      registeredModel.setName(spec.getName());
      registeredModel.setDescription(spec.getDescription());
      registeredModelService.insert(registeredModel);
      return new JsonResponse.Builder<String>(Response.Status.OK).success(true)
        .message("Create a registered model instance").build();
    } catch (SubmarineRuntimeException e) {
      return parseRegisteredModelServiceException(e);
    }
  }

  /**
   * List all registered models.
   *
   * @return registered model list
   */
  @GET
  @Operation(summary = "List registered models", tags = { "registered-model" }, responses = {
      @ApiResponse(description = "successful operation",
      content = @Content(schema = @Schema(implementation = JsonResponse.class))) })
  public Response listRegisteredModels() {
    try {
      List<RegisteredModelEntity> registeredModelList = registeredModelService.selectAll();
      return new JsonResponse.Builder<List<RegisteredModelEntity>>(Response.Status.OK).success(true)
        .message("List all registered model instances").result(registeredModelList).build();
    } catch (SubmarineRuntimeException e) {
      return parseRegisteredModelServiceException(e);
    }
  }

  /**
   * Get detailed info about the registered model by registered model name.
   *
   * @param name registered model name
   * @return detailed info about the registered model
   */
  @GET
  @Path("/{name}")
  @Operation(summary = "Get detailed info about the registered model",
      tags = { "registered-model" }, responses = {
      @ApiResponse(description = "successful operation",
          content = @Content(schema = @Schema(implementation = JsonResponse.class))),
      @ApiResponse(responseCode = "404", description = "RegisteredModelEntity not found") })
  public Response getRegisteredModel(@PathParam(RestConstants.REGISTERED_MODEL_NAME) String name) {
    try {
      RegisteredModelEntity registeredModel = registeredModelService.selectWithTag(name);
      return new JsonResponse.Builder<RegisteredModelEntity>(Response.Status.OK).success(true)
        .message("Get the registered model instance").result(registeredModel).build();
    } catch (SubmarineRuntimeException e) {
      return parseRegisteredModelServiceException(e);
    }
  }

  /**
   * Delete the registered model with registered model name.
   *
   * @param name registered model name
   * @return success message
   */
  @DELETE
  @Path("/{name}")
  @Operation(summary = "Delete the registered model", tags = { "registered-model" }, responses = {
      @ApiResponse(description = "successful operation",
        content = @Content(schema = @Schema(implementation = JsonResponse.class))),
      @ApiResponse(responseCode = "404", description = "RegisteredModelEntity not found") })
  public Response deleteRegisteredModel(@PathParam(RestConstants.REGISTERED_MODEL_NAME) String name) {
    try {
      registeredModelService.delete(name);
      return new JsonResponse.Builder<String>(Response.Status.OK).success(true)
        .message("Delete the registered model instance").build();
    } catch (SubmarineRuntimeException e) {
      return parseRegisteredModelServiceException(e);
    }
  }

  private Response parseRegisteredModelServiceException(SubmarineRuntimeException e) {
    return new JsonResponse.Builder<String>(e.getCode()).message(e.getMessage()).build();
  }

  /**
   * Check if register model spec is valid spec.
   *
   * @param spec register model spec
   */
  private void checkRegisteredModelSpec(RegisteredModelSpec spec) {
    if (spec == null) {
      throw new SubmarineRuntimeException(Response.Status.OK.getStatusCode(),
        "Invalid. RegisteredModel Spec object is null.");
    }
    if (spec.getName() == null || spec.getName() == "") {
      throw new SubmarineRuntimeException(Response.Status.OK.getStatusCode(),
        "Invalid. RegisteredModel name is null.");
    }
    if (spec.getDescription() == null) {
      spec.setDescription("");
    }
    List<RegisteredModelEntity> registeredModels = registeredModelService.selectAll();
    for (RegisteredModelEntity registerModel : registeredModels) {
      if (registerModel.getName().equals(spec.getName())) {
        throw new SubmarineRuntimeException(Response.Status.OK.getStatusCode(),
          "Invalid. RegisteredModel with same name is already existed.");
      }
    }
  }
}
