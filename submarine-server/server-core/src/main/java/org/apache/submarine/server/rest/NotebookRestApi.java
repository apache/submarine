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

import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.responses.ApiResponse;

import org.apache.submarine.commons.utils.exception.SubmarineRuntimeException;
import org.apache.submarine.server.api.notebook.Notebook;
import org.apache.submarine.server.api.spec.NotebookSpec;
import org.apache.submarine.server.notebook.NotebookManager;
import org.apache.submarine.server.utils.response.JsonResponse;

import javax.ws.rs.Consumes;
import javax.ws.rs.DELETE;
import javax.ws.rs.GET;
import javax.ws.rs.POST;
import javax.ws.rs.Path;
import javax.ws.rs.PathParam;
import javax.ws.rs.Produces;
import javax.ws.rs.QueryParam;
import javax.ws.rs.core.MediaType;
import javax.ws.rs.core.Response;
import java.util.List;

/**
 * Notebook REST API v1. It can accept {@link NotebookSpec} to create a notebook server.
 */
@Path(RestConstants.V1 + "/" + RestConstants.NOTEBOOK)
@Produces({MediaType.APPLICATION_JSON + "; " + RestConstants.CHARSET_UTF8})
public class NotebookRestApi {

  /* Notebook manager  */
  private final NotebookManager notebookManager = NotebookManager.getInstance();

  /**
   * Return the Pong message for test the connectivity
   * @return Pong message
   */
  @GET
  @Path(RestConstants.PING)
  @Consumes(MediaType.APPLICATION_JSON)
  @Operation(summary = "Ping submarine server",
          tags = {"notebook"},
          description = "Return the Pong message for test the connectivity",
          responses = {
                  @ApiResponse(responseCode = "200", description = "successful operation",
                          content = @Content(schema = @Schema(implementation = String.class)))})
  public Response ping() {
    return new JsonResponse.Builder<String>(Response.Status.OK)
            .success(true).result("Pong").build();
  }

  /**
   * Create a notebook with spec
   * @param spec notebook spec
   * @return the detailed info about created notebook
   */
  @POST
  @Consumes({RestConstants.MEDIA_TYPE_YAML, MediaType.APPLICATION_JSON})
  @Operation(
          summary = "Create a notebook instance",
          tags = {"notebook"},
          responses = {
                  @ApiResponse(description = "successful operation", content = @Content(
                          schema = @Schema(implementation = JsonResponse.class)))})
  public Response createNotebook(NotebookSpec spec) {
    try {
      Notebook notebook = notebookManager.createNotebook(spec);
      return new JsonResponse.Builder<Notebook>(Response.Status.OK).success(true)
              .message("Create a notebook instance").result(notebook).build();
    } catch (SubmarineRuntimeException e) {
      return parseNotebookServiceException(e);
    }
  }

  /**
   * List all notebooks
   * @param id user id
   * @return notebook list
   */
  @GET
  @Operation(
          summary = "List notebooks",
          tags = {"notebook"},
          responses = {
                  @ApiResponse(description = "successful operation", content = @Content(
                          schema = @Schema(implementation = JsonResponse.class)))})
  public Response listNotebooks(@QueryParam("id") String id) {
    try {
      List<Notebook> notebookList = notebookManager.listNotebooksByUserId(id);
      return new JsonResponse.Builder<List<Notebook>>(Response.Status.OK).success(true)
              .message("List all notebook instances").result(notebookList).build();
    } catch (SubmarineRuntimeException e) {
      return parseNotebookServiceException(e);
    }
  }

  /**
   * Get detailed info about the notebook by notebook id
   * @param id notebook id
   * @return detailed info about the notebook
   */
  @GET
  @Path("/{id}")
  @Operation(
          summary = "Get detailed info about the notebook",
          tags = {"notebook"},
          responses = {
                  @ApiResponse(
                          description = "successful operation", content = @Content(
                          schema = @Schema(implementation = JsonResponse.class))),
                  @ApiResponse(responseCode = "404", description = "Notebook not found")})
  public Response getNotebook(@PathParam(RestConstants.NOTEBOOK_ID) String id) {
    try {
      Notebook notebook = notebookManager.getNotebook(id);
      return new JsonResponse.Builder<Notebook>(Response.Status.OK).success(true)
              .message("Get the notebook instance").result(notebook).build();
    } catch (SubmarineRuntimeException e) {
      return parseNotebookServiceException(e);
    }
  }

  /**
   * Delete the notebook with notebook id
   * @param id notebook id
   * @return the detailed info about deleted notebook
   */
  @DELETE
  @Path("/{id}")
  @Operation(
          summary = "Delete the notebook",
          tags = {"notebook"},
          responses = {
                  @ApiResponse(
                          description = "successful operation", content = @Content(
                          schema = @Schema(implementation = JsonResponse.class))),
                  @ApiResponse(responseCode = "404", description = "Notebook not found")})
  public Response deleteNotebook(@PathParam(RestConstants.NOTEBOOK_ID) String id) {
    try {
      Notebook notebook = notebookManager.deleteNotebook(id);
      return new JsonResponse.Builder<Notebook>(Response.Status.OK).success(true)
              .message("Delete the notebook instance").result(notebook).build();
    } catch (SubmarineRuntimeException e) {
      return parseNotebookServiceException(e);
    }
  }

  private Response parseNotebookServiceException(SubmarineRuntimeException e) {
    return new JsonResponse.Builder<String>(e.getCode()).message(e.getMessage()).build();
  }

}
