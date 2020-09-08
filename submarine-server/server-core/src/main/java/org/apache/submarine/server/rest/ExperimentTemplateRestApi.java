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
import org.apache.submarine.server.api.experimenttemplate.ExperimentTemplate;
import org.apache.submarine.server.api.spec.ExperimentTemplateSpec;
import org.apache.submarine.server.experimenttemplate.ExperimentTemplateManager;
import org.apache.submarine.server.response.JsonResponse;

import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.responses.ApiResponse;

/**
 * ExperimentTemplate REST API v1. It can accept {@link ExperimentTemplateSpec} to create a
 * experimentTemplate.
 */
@Path(RestConstants.V1 + "/" + RestConstants.EXPERIMENT_TEMPLATES)
@Produces({MediaType.APPLICATION_JSON + "; " + RestConstants.CHARSET_UTF8})
public class ExperimentTemplateRestApi {
  private final ExperimentTemplateManager experimentTemplateManager =
      ExperimentTemplateManager.getInstance();

  /**
   * Returns the contents of {@link experimentTemplate}.
   * @param spec experimentTemplate spec
   * @return the contents of experimentTemplate
   */
  @POST
  @Consumes({RestConstants.MEDIA_TYPE_YAML, MediaType.APPLICATION_JSON})
  @Operation(summary = "Create a experimentTemplate",
          tags = {"experimentTemplate"},
          responses = {
                  @ApiResponse(description = "successful operation", 
                      content = @Content(
                          schema = @Schema(
                              implementation = JsonResponse.class)))})
  public Response createExperimentTemplate(ExperimentTemplateSpec spec) {
    try {
      ExperimentTemplate experimentTemplate = experimentTemplateManager.createExperimentTemplate(spec);
      return new JsonResponse.Builder<ExperimentTemplate>(Response.Status.OK)
          .success(true).result(experimentTemplate).build();
    } catch (SubmarineRuntimeException e) {
      return parseExperimentTemplateServiceException(e);
    }
  }
  
  /**
   * Update experimentTemplate.
   * @param name Name of the experimentTemplate
   * @param spec experimentTemplate spec
   * @return the detailed info about updated experimentTemplate
   */
  @PATCH
  @Path("/{id}")
  @Consumes({RestConstants.MEDIA_TYPE_YAML, MediaType.APPLICATION_JSON})
  @Operation(summary = "Update the experimentTemplate with job spec",
          tags = {"experimentTemplates"},
          responses = {
                  @ApiResponse(description = "successful operation", 
                      content = @Content(
                          schema = @Schema(
                              implementation = JsonResponse.class))),
                  @ApiResponse(
                      responseCode = "404", 
                      description = "ExperimentTemplate not found")})
  public Response updateExperimentTemplate(
      @PathParam(RestConstants.EXPERIMENT_TEMPLATE_ID) String name,
      ExperimentTemplateSpec spec) {
    try {
      ExperimentTemplate experimentTemplate =
          experimentTemplateManager.updateExperimentTemplate(name, spec);
      return new JsonResponse.Builder<ExperimentTemplate>(Response.Status.OK)
          .success(true).result(experimentTemplate).build();
    } catch (SubmarineRuntimeException e) {
      return parseExperimentTemplateServiceException(e);
    }
  }

  /**
   * Returns the experimentTemplate that deleted.
   * @param name Name of the experimentTemplate
   * @return the detailed info about deleted experimentTemplate
   */
  @DELETE
  @Path("/{id}")
  @Operation(summary = "Delete the experimentTemplate",
          tags = {"experimentTemplates"},
          responses = {
                  @ApiResponse(description = "successful operation", 
                      content = @Content(
                          schema = @Schema(implementation = JsonResponse.class))),
                  @ApiResponse(
                      responseCode = "404", description = "ExperimentTemplate not found")})
  public Response deleteExperimentTemplate(
      @PathParam(RestConstants.EXPERIMENT_TEMPLATE_ID) String name) {
    try {
      ExperimentTemplate experimentTemplate = experimentTemplateManager.deleteExperimentTemplate(name);
      return new JsonResponse.Builder<ExperimentTemplate>(Response.Status.OK)
          .success(true).result(experimentTemplate).build();
    } catch (SubmarineRuntimeException e) {
      return parseExperimentTemplateServiceException(e);
    }
  }
  
  /**
   * List all experimentTemplates.
   * @return experimentTemplate list
   */
  @GET
  @Operation(summary = "List of ExperimentTemplates",
          tags = {"experimentTemplates"},
          responses = {
                  @ApiResponse(description = "successful operation", 
                      content = @Content(
                          schema = @Schema(
                              implementation = JsonResponse.class)))})
  public Response listExperimentTemplate(@QueryParam("status") String status) {
    try {
      List<ExperimentTemplate> experimentTemplateList =
          experimentTemplateManager.listExperimentTemplates(status);
      return new JsonResponse.Builder<List<ExperimentTemplate>>(Response.Status.OK)
          .success(true).result(experimentTemplateList).build();
    } catch (SubmarineRuntimeException e) {
      return parseExperimentTemplateServiceException(e);
    }
  }

  /**
   * Returns details for the given experimentTemplate.
   * @param name Name of the experimentTemplate
   * @return the contents of experimentTemplate
   */
  @GET
  @Path("/{id}")
  @Operation(summary = "Find experimentTemplate by name",
          tags = {"experimentTemplate"},
          responses = {
                  @ApiResponse(description = "successful operation", 
                      content = @Content(
                          schema = @Schema(implementation = JsonResponse.class))),
                  @ApiResponse(
                      responseCode = "404", 
                      description = "ExperimentTemplate not found")})
  public Response getExperimentTemplate(
      @PathParam(RestConstants.EXPERIMENT_TEMPLATE_ID) String name) {
    try {
      ExperimentTemplate experimentTemplate = experimentTemplateManager.getExperimentTemplate(name);
      return new JsonResponse.Builder<ExperimentTemplate>(Response.Status.OK)
          .success(true).result(experimentTemplate).build();
    } catch (SubmarineRuntimeException e) {
      return parseExperimentTemplateServiceException(e);
    }
  }

  private Response parseExperimentTemplateServiceException(
      SubmarineRuntimeException e) {
    return new JsonResponse.Builder<String>(e.getCode()).message(e.getMessage())
        .build();
  }
}
