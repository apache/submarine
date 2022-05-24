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
import org.apache.submarine.server.database.model.entities.RegisteredModelEntity;

import org.apache.submarine.server.manager.RegisteredModelManager;
import org.apache.submarine.server.utils.response.JsonResponse;


/**
 * Registered model REST API v1.
 */
@Path(RestConstants.V1 + "/" + RestConstants.REGISTERED_MODEL)
@Produces({ MediaType.APPLICATION_JSON + "; " + RestConstants.CHARSET_UTF8 })
public class RegisteredModelRestApi {

  /* Registered model manager  */
  private final RegisteredModelManager registeredModelManager = RegisteredModelManager.getInstance();

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
   * @param entity registered model entity
   * example: {
   *   'name': 'example_name'
   *   'description': 'example_description'
   *   'tags': ['123', '456']
   * }
   * @return success message
   */
  @POST
  @Consumes({ RestConstants.MEDIA_TYPE_YAML, MediaType.APPLICATION_JSON })
  @Operation(summary = "Create a registered model instance", tags = { "registered-model" }, responses = {
      @ApiResponse(description = "successful operation",
          content = @Content(schema = @Schema(implementation = JsonResponse.class))) })
  public Response createRegisteredModel(RegisteredModelEntity entity) {
    try {
      registeredModelManager.createRegisteredModel(entity);
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
      List<RegisteredModelEntity> registeredModelList = registeredModelManager.listRegisteredModels();
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
      RegisteredModelEntity registeredModel = registeredModelManager.getRegisteredModel(name);
      return new JsonResponse.Builder<RegisteredModelEntity>(Response.Status.OK).success(true)
        .message("Get the registered model instance").result(registeredModel).build();
    } catch (SubmarineRuntimeException e) {
      return parseRegisteredModelServiceException(e);
    }
  }

  /**
   * Update the registered model with registered model name.
   *
   * @param name   old registered model name
   * @param entity registered model entity
   * example: {
   *   'name': 'new_name'
   *   'description': 'new_description'
   * }
   * @return success message
   */
  @PATCH
  @Path("/{name}")
  @Operation(summary = "Update the registered model", tags = { "registered-model" }, responses = {
      @ApiResponse(description = "successful operation",
          content = @Content(schema = @Schema(implementation = JsonResponse.class))),
      @ApiResponse(responseCode = "404", description = "RegisteredModelEntity not found") })
  public Response updateRegisteredModel(
      @PathParam(RestConstants.REGISTERED_MODEL_NAME) String name, RegisteredModelEntity entity) {
    try {
      registeredModelManager.updateRegisteredModel(name, entity);
      return new JsonResponse.Builder<String>(Response.Status.OK).success(true)
        .message("Update the registered model instance").build();
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
      @ApiResponse(responseCode = "404", description = "RegisteredModelEntity not found"),
      @ApiResponse(responseCode = "406", description = "Some version of models are in the production stage"),
      @ApiResponse(responseCode = "500", description = "Some error happen in server")})
  public Response deleteRegisteredModel(@PathParam(RestConstants.REGISTERED_MODEL_NAME) String name) {
    try {
      registeredModelManager.deleteRegisteredModel(name);
      return new JsonResponse.Builder<String>(Response.Status.OK).success(true)
        .message("Delete the registered model instance").build();
    } catch (SubmarineRuntimeException e) {
      return parseRegisteredModelServiceException(e);
    }
  }

  /**
   * Create a registered model tag.
   *
   * @param name registered model name
   * @param tag  tag name
   * @return success message
   */
  @POST
  @Path("/tag")
  @Consumes({ RestConstants.MEDIA_TYPE_YAML, MediaType.APPLICATION_JSON })
  @Operation(summary = "Create a registered model tag instance", tags = { "registered-model" }, responses = {
      @ApiResponse(description = "successful operation",
          content = @Content(schema = @Schema(implementation = JsonResponse.class))) })
  public Response createRegisteredModelTag(@DefaultValue("") @QueryParam("name") String name,
                                           @DefaultValue("") @QueryParam("tag") String tag) {
    try {
      registeredModelManager.createRegisteredModelTag(name, tag);
      return new JsonResponse.Builder<String>(Response.Status.OK).success(true)
          .message("Create a registered model tag instance").build();
    } catch (SubmarineRuntimeException e) {
      return parseRegisteredModelServiceException(e);
    }
  }

  /**
   * Delete a registered model tag.
   *
   * @param name registered model name
   * @param tag  tag name
   * @return success message
   */
  @DELETE
  @Path("/tag")
  @Consumes({ RestConstants.MEDIA_TYPE_YAML, MediaType.APPLICATION_JSON })
  @Operation(summary = "Delete a registered model tag instance", tags = { "registered-model" }, responses = {
      @ApiResponse(description = "successful operation",
          content = @Content(schema = @Schema(implementation = JsonResponse.class))) })
  public Response deleteRegisteredModelTag(@DefaultValue("") @QueryParam("name") String name,
                                           @DefaultValue("") @QueryParam("tag") String tag) {
    try {
      registeredModelManager.deleteRegisteredModelTag(name, tag);
      return new JsonResponse.Builder<String>(Response.Status.OK).success(true)
          .message("Delete a registered model tag instance").build();
    } catch (SubmarineRuntimeException e) {
      return parseRegisteredModelServiceException(e);
    }
  }

  private Response parseRegisteredModelServiceException(SubmarineRuntimeException e) {
    return new JsonResponse.Builder<String>(e.getCode()).message(e.getMessage()).build();
  }
}
