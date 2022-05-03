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
import javax.ws.rs.PATCH;
import javax.ws.rs.POST;
import javax.ws.rs.Path;
import javax.ws.rs.PathParam;
import javax.ws.rs.Produces;
import javax.ws.rs.QueryParam;
import javax.ws.rs.core.MediaType;
import javax.ws.rs.core.Response;

import org.apache.submarine.commons.utils.exception.SubmarineRuntimeException;
import org.apache.submarine.server.api.environment.Environment;
import org.apache.submarine.server.api.spec.EnvironmentSpec;
import org.apache.submarine.server.manager.EnvironmentManager;
import org.apache.submarine.server.utils.response.JsonResponse;

import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.responses.ApiResponse;

/**
 * Environment REST API v1. It can accept {@link EnvironmentSpec} to create a
 * environment.
 */
@Path(RestConstants.V1 + "/" + RestConstants.ENVIRONMENT)
@Produces({MediaType.APPLICATION_JSON + "; " + RestConstants.CHARSET_UTF8})
public class EnvironmentRestApi {
  private final EnvironmentManager environmentManager =
      EnvironmentManager.getInstance();

  /**
   * Returns the contents of {@link environment}.
   * @param spec environment spec
   * @return the contents of environment
   */
  @POST
  @Consumes({RestConstants.MEDIA_TYPE_YAML, MediaType.APPLICATION_JSON})
  @Operation(summary = "Create a environment",
          tags = {"environment"},
          responses = {
                  @ApiResponse(description = "successful operation",
                      content = @Content(
                          schema = @Schema(
                              implementation = JsonResponse.class)))})
  public Response createEnvironment(EnvironmentSpec spec) {
    try {
      Environment environment = environmentManager.createEnvironment(spec);
      return new JsonResponse.Builder<Environment>(Response.Status.OK)
          .success(true).result(environment).build();
    } catch (SubmarineRuntimeException e) {
      return parseEnvironmentServiceException(e);
    }
  }

  /**
   * Update environment.
   * @param name Name of the environment
   * @param spec environment spec
   * @return the detailed info about updated environment
   */
  @PATCH
  @Path("/{id}")
  @Consumes({RestConstants.MEDIA_TYPE_YAML, MediaType.APPLICATION_JSON})
  @Operation(summary = "Update the environment with job spec",
          tags = {"environment"},
          responses = {
                  @ApiResponse(description = "successful operation",
                      content = @Content(
                          schema = @Schema(
                              implementation = JsonResponse.class))),
                  @ApiResponse(
                      responseCode = "404",
                      description = "Environment not found")})
  public Response updateEnvironment(
      @PathParam(RestConstants.ENVIRONMENT_ID) String name,
      EnvironmentSpec spec) {
    try {
      Environment environment =
          environmentManager.updateEnvironment(name, spec);
      return new JsonResponse.Builder<Environment>(Response.Status.OK)
          .success(true).result(environment).build();
    } catch (SubmarineRuntimeException e) {
      return parseEnvironmentServiceException(e);
    }
  }

  /**
   * Returns the environment that deleted.
   * @param name Name of the environment
   * @return the detailed info about deleted environment
   */
  @DELETE
  @Path("/{id}")
  @Operation(summary = "Delete the environment",
          tags = {"environment"},
          responses = {
                  @ApiResponse(description = "successful operation",
                      content = @Content(
                          schema = @Schema(implementation = JsonResponse.class))),
                  @ApiResponse(
                      responseCode = "404", description = "Environment not found")})
  public Response deleteEnvironment(
      @PathParam(RestConstants.ENVIRONMENT_ID) String name) {
    try {
      Environment environment = environmentManager.deleteEnvironment(name);
      return new JsonResponse.Builder<Environment>(Response.Status.OK)
          .success(true).result(environment).build();
    } catch (SubmarineRuntimeException e) {
      return parseEnvironmentServiceException(e);
    }
  }

  /**
   * List all environments.
   * @return environment list
   */
  @GET
  @Operation(summary = "List of Environments",
          tags = {"environment"},
          responses = {
                  @ApiResponse(description = "successful operation",
                      content = @Content(
                          schema = @Schema(
                              implementation = JsonResponse.class)))})
  public Response listEnvironment(@QueryParam("status") String status) {
    try {
      List<Environment> environmentList =
          environmentManager.listEnvironments(status);
      return new JsonResponse.Builder<List<Environment>>(Response.Status.OK)
          .success(true).result(environmentList).build();
    } catch (SubmarineRuntimeException e) {
      return parseEnvironmentServiceException(e);
    }
  }

  /**
   * Returns details for the given environment.
   * @param name Name of the environment
   * @return the contents of environment
   */
  @GET
  @Path("/{id}")
  @Operation(summary = "Find environment by name",
          tags = {"environment"},
          responses = {
                  @ApiResponse(description = "successful operation",
                      content = @Content(
                          schema = @Schema(implementation = JsonResponse.class))),
                  @ApiResponse(
                      responseCode = "404",
                      description = "Environment not found")})
  public Response getEnvironment(
      @PathParam(RestConstants.ENVIRONMENT_ID) String name) {
    try {
      Environment environment = environmentManager.getEnvironment(name);
      return new JsonResponse.Builder<Environment>(Response.Status.OK)
          .success(true).result(environment).build();
    } catch (SubmarineRuntimeException e) {
      return parseEnvironmentServiceException(e);
    }
  }

  private Response parseEnvironmentServiceException(
      SubmarineRuntimeException e) {
    return new JsonResponse.Builder<String>(e.getCode()).message(e.getMessage())
        .build();
  }
}
