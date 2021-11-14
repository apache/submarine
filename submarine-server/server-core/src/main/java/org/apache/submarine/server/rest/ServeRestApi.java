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
import javax.ws.rs.GET;
import javax.ws.rs.POST;
import javax.ws.rs.Path;
import javax.ws.rs.Produces;
import javax.ws.rs.core.MediaType;
import javax.ws.rs.core.Response;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.responses.ApiResponse;

import org.apache.submarine.commons.utils.exception.SubmarineRuntimeException;
import org.apache.submarine.server.api.model.ServeResponse;
import org.apache.submarine.server.api.model.ServeSpec;
import org.apache.submarine.server.model.ModelManager;
import org.apache.submarine.server.response.JsonResponse;



@Path(RestConstants.V1 + "/" + RestConstants.SERVE)
@Produces({MediaType.APPLICATION_JSON + "; " + RestConstants.CHARSET_UTF8})
public class ServeRestApi {

  private final ModelManager modelManager = ModelManager.getInstance();

  /**
   * Return the Pong message for test the connectivity.
   * @return Pong message
   */
  @GET
  @Path(RestConstants.PING)
  @Consumes(MediaType.APPLICATION_JSON)
  @Operation(summary = "Ping submarine server",
             tags = {"serve"},
             description = "Return the Pong message for test the connectivity",
             responses = {@ApiResponse(responseCode = "200", description = "successful operation",
             content = @Content(schema = @Schema(implementation = String.class)))})
  public Response ping() {
    return new JsonResponse.Builder<String>(Response.Status.OK)
            .success(true).result("Pong").build();
  }

  @POST
  @Consumes({ RestConstants.MEDIA_TYPE_YAML, MediaType.APPLICATION_JSON })
  @Operation(summary = "Create a model serve instance", tags = { "serve" }, responses = {
          @ApiResponse(description = "successful operation",
                  content = @Content(schema = @Schema(implementation = JsonResponse.class))) })
  public Response createServe(ServeSpec spec) {
    try {
      ServeResponse serveResponse = modelManager.createServe(spec);
      return new JsonResponse.Builder<ServeResponse>(Response.Status.OK).success(true)
              .message("Create a model serve instance").result(serveResponse).build();
    } catch (SubmarineRuntimeException e) {
      return parseModelVersionServiceException(e);
    }
  }

  private Response parseModelVersionServiceException(SubmarineRuntimeException e) {
    return new JsonResponse.Builder<String>(e.getCode()).message(e.getMessage()).build();
  }
}
