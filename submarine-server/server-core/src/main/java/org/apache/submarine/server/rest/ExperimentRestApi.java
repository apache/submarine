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
import java.util.List;

import com.google.common.annotations.VisibleForTesting;

import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.responses.ApiResponse;

import org.apache.submarine.commons.utils.exception.SubmarineRuntimeException;
import org.apache.submarine.server.api.experiment.Experiment;
import org.apache.submarine.server.api.experiment.TensorboardInfo;
import org.apache.submarine.server.api.experiment.MlflowInfo;
import org.apache.submarine.server.experiment.ExperimentManager;
import org.apache.submarine.server.experimenttemplate.ExperimentTemplateManager;
import org.apache.submarine.server.api.experiment.ExperimentLog;
import org.apache.submarine.server.api.experimenttemplate.ExperimentTemplateSubmit;
import org.apache.submarine.server.api.spec.ExperimentSpec;
import org.apache.submarine.server.response.JsonResponse;
import org.apache.submarine.server.s3.Client;

/**
 * Experiment Service REST API v1
 */
@Path(RestConstants.V1 + "/" + RestConstants.EXPERIMENT)
@Produces({MediaType.APPLICATION_JSON + "; " + RestConstants.CHARSET_UTF8})
public class ExperimentRestApi {
  private ExperimentManager experimentManager = ExperimentManager.getInstance();
  private Client minioClient = new Client();

  @VisibleForTesting
  public void setExperimentManager(ExperimentManager experimentManager) {
    this.experimentManager = experimentManager;
  }

  /**
   * Return the Pong message for test the connectivity
   *
   * @return Pong message
   */
  @GET
  @Path(RestConstants.PING)
  @Consumes(MediaType.APPLICATION_JSON)
  @Operation(summary = "Ping submarine server",
      tags = {"experiment"},
      description = "Return the Pong message for test the connectivity",
      responses = {
          @ApiResponse(responseCode = "200", description = "successful operation",
              content = @Content(schema = @Schema(implementation = String.class)))})
  public Response ping() {
    return new JsonResponse.Builder<String>(Response.Status.OK)
        .success(true).result("Pong").build();
  }


  /**
   * Returns the contents of {@link Experiment} that submitted by user.
   *
   * @param spec spec
   * @return the contents of experiment
   */

  @POST
  @Consumes({RestConstants.MEDIA_TYPE_YAML, MediaType.APPLICATION_JSON})
  @Operation(summary = "Create an experiment",
      tags = {"experiment"},
      responses = {
          @ApiResponse(description = "successful operation", content = @Content(
              schema = @Schema(implementation = JsonResponse.class)))})
  public Response createExperiment(ExperimentSpec spec) {
    try {
      Experiment experiment = experimentManager.createExperiment(spec);
      return new JsonResponse.Builder<Experiment>(Response.Status.OK).success(true)
          .result(experiment).build();
    } catch (SubmarineRuntimeException e) {
      return parseExperimentServiceException(e);
    }
  }

  /**
   * Returns the contents of {@link Experiment} that submitted by user.
   *
   * @param name template name
   * @param spec
   * @return the contents of experiment
   */
  @POST
  @Path("/{name}")
  @Consumes({RestConstants.MEDIA_TYPE_YAML, MediaType.APPLICATION_JSON})
  @Operation(summary = "use experiment template to create an experiment",
      tags = {"experiment"},
      responses = {
          @ApiResponse(description = "successful operation", content = @Content(
              schema = @Schema(implementation = JsonResponse.class)))})
  public Response SubmitExperimentTemplate(@PathParam("name") String name,
        ExperimentTemplateSubmit spec) {
    try {
      spec.setName(name);

      Experiment experiment = ExperimentTemplateManager.getInstance().submitExperimentTemplate(spec);
      return new JsonResponse.Builder<Experiment>(Response.Status.OK)
          .success(true).result(experiment).build();
    } catch (SubmarineRuntimeException e) {
      return parseExperimentServiceException(e);
    }
  }

  /**
   * List all experiment for the user
   *
   * @return experiment list
   */
  @GET
  @Operation(summary = "List experiments",
      tags = {"experiment"},
      responses = {
          @ApiResponse(description = "successful operation", content = @Content(
              schema = @Schema(implementation = JsonResponse.class)))})
  public Response listExperiments(@QueryParam("status") String status) {
    try {
      List<Experiment> experimentList = experimentManager.listExperimentsByStatus(status);
      return new JsonResponse.Builder<List<Experiment>>(Response.Status.OK).success(true)
          .result(experimentList).build();
    } catch (SubmarineRuntimeException e) {
      return parseExperimentServiceException(e);
    }
  }

  /**
   * Returns the experiment detailed info by specified experiment id
   *
   * @param id experiment id
   * @return the detailed info of experiment
   */
  @GET
  @Path("/{id}")
  @Operation(summary = "Get the experiment's detailed info by id",
      tags = {"experiment"},
      responses = {
          @ApiResponse(description = "successful operation", content = @Content(
              schema = @Schema(implementation = JsonResponse.class))),
          @ApiResponse(responseCode = "404", description = "Experiment not found")})
  public Response getExperiment(@PathParam(RestConstants.ID) String id) {
    try {
      Experiment experiment = experimentManager.getExperiment(id);
      return new JsonResponse.Builder<Experiment>(Response.Status.OK).success(true)
          .result(experiment).build();
    } catch (SubmarineRuntimeException e) {
      return parseExperimentServiceException(e);
    }
  }

  @PATCH
  @Path("/{id}")
  @Consumes({RestConstants.MEDIA_TYPE_YAML, MediaType.APPLICATION_JSON})
  @Operation(summary = "Update the experiment in the submarine server with spec",
      tags = {"experiment"},
      responses = {
          @ApiResponse(description = "successful operation", content = @Content(
              schema = @Schema(implementation = JsonResponse.class))),
          @ApiResponse(responseCode = "404", description = "Experiment not found")})
  public Response patchExperiment(@PathParam(RestConstants.ID) String id, ExperimentSpec spec) {
    try {
      Experiment experiment = experimentManager.patchExperiment(id, spec);
      return new JsonResponse.Builder<Experiment>(Response.Status.OK).success(true)
          .result(experiment).build();
    } catch (SubmarineRuntimeException e) {
      return parseExperimentServiceException(e);
    }
  }

  /**
   * Returns the experiment that deleted
   *
   * @param id experiment id
   * @return the detailed info about deleted experiment
   */
  @DELETE
  @Path("/{id}")
  @Operation(summary = "Delete the experiment",
      tags = {"experiment"},
      responses = {
          @ApiResponse(description = "successful operation", content = @Content(
              schema = @Schema(implementation = JsonResponse.class))),
          @ApiResponse(responseCode = "404", description = "Experiment not found")})
  public Response deleteExperiment(@PathParam(RestConstants.ID) String id) {
    Experiment experiment;
    try {
      experiment = experimentManager.deleteExperiment(id);
    } catch (SubmarineRuntimeException e) {
      return parseExperimentServiceException(e);
    }
    return new JsonResponse.Builder<Experiment>(Response.Status.OK).success(true)
          .result(experiment).build();
  }

  @GET
  @Path("/logs")
  @Operation(summary = "List experiment's log",
      tags = {"experiment"},
      responses = {
          @ApiResponse(description = "successful operation", content = @Content(
              schema = @Schema(implementation = JsonResponse.class))),
      })
  public Response listLog(@QueryParam("status") String status) {
    try {
      List<ExperimentLog> experimentLogList = experimentManager.listExperimentLogsByStatus(status);
      return new JsonResponse.Builder<List<ExperimentLog>>(Response.Status.OK).success(true)
          .result(experimentLogList).build();

    } catch (SubmarineRuntimeException e) {
      return parseExperimentServiceException(e);
    }
  }

  @GET
  @Path("/logs/{id}")
  @Operation(summary = "Log experiment by id",
      tags = {"experiment"},
      responses = {
          @ApiResponse(description = "successful operation", content = @Content(
              schema = @Schema(implementation = JsonResponse.class))),
          @ApiResponse(responseCode = "404", description = "Experiment not found")})
  public Response getLog(@PathParam(RestConstants.ID) String id) {
    try {
      ExperimentLog experimentLog = experimentManager.getExperimentLog(id);
      return new JsonResponse.Builder<ExperimentLog>(Response.Status.OK).success(true)
          .result(experimentLog).build();

    } catch (SubmarineRuntimeException e) {
      return parseExperimentServiceException(e);
    }
  }

  @GET
  @Path("/artifacts/{id}")
  @Operation(summary = "List artifact paths by id",
      tags = {"experiment"},
      responses = {
          @ApiResponse(description = "successful operation", content = @Content(
              schema = @Schema(implementation = JsonResponse.class))),
          @ApiResponse(responseCode = "404", description = "Experiment not found")})
  public Response getArtifactPaths(@PathParam(RestConstants.ID) String id) {
    try {
      List<String> artifactPaths = minioClient.listArtifactByExperimentId(id);
      return new JsonResponse.Builder<List<String>>(Response.Status.OK).success(true)
          .result(artifactPaths).build();

    } catch (SubmarineRuntimeException e) {
      return parseExperimentServiceException(e);
    }
  }

  @GET
  @Path("/tensorboard")
  @Operation(summary = "Get tensorboard's information",
      tags = {"experiment"},
      responses = {
        @ApiResponse(description = "successful operation", content = @Content(
          schema = @Schema(implementation = JsonResponse.class))),
        @ApiResponse(responseCode = "404", description = "Tensorboard not found")})
  public Response getTensorboardInfo() {
    try {
      TensorboardInfo tensorboardInfo = experimentManager.getTensorboardInfo();
      return new JsonResponse.Builder<TensorboardInfo>(Response.Status.OK).success(true)
        .result(tensorboardInfo).build();
    } catch (SubmarineRuntimeException e) {
      return parseExperimentServiceException(e);
    }
  }

  @GET
  @Path("/mlflow")
  @Operation(summary = "Get mlflow's information",
          tags = {"experiment"},
          responses = {
                  @ApiResponse(description = "successful operation", content = @Content(
                          schema = @Schema(implementation = JsonResponse.class))),
                  @ApiResponse(responseCode = "404", description = "MLflow not found")})
  public Response getMLflowInfo() {
    try {
      MlflowInfo mlflowInfo = experimentManager.getMLflowInfo();
      return new JsonResponse.Builder<MlflowInfo>(Response.Status.OK).success(true)
              .result(mlflowInfo).build();
    } catch (SubmarineRuntimeException e) {
      return parseExperimentServiceException(e);
    }
  }

  private Response parseExperimentServiceException(SubmarineRuntimeException e) {
    return new JsonResponse.Builder<String>(e.getCode())
      .message(e.getMessage().equals("Conflict") ? "Duplicated experiment name" : e.getMessage()).build();
  }
}
