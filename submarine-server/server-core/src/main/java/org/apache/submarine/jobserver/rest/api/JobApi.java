/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 * <p>
 * http://www.apache.org/licenses/LICENSE-2.0
 * <p>
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.submarine.jobserver.rest.api;

import org.apache.submarine.jobserver.rest.dao.JsonResponse;
import org.apache.submarine.jobserver.rest.dao.MLJobSpec;
import org.apache.submarine.jobserver.rest.dao.RestConstants;

import javax.ws.rs.Consumes;
import javax.ws.rs.DELETE;
import javax.ws.rs.GET;
import javax.ws.rs.POST;
import javax.ws.rs.Path;
import javax.ws.rs.PathParam;
import javax.ws.rs.Produces;
import javax.ws.rs.core.MediaType;
import javax.ws.rs.core.Response;

/**
 * ML job rest API v1. It can accept MLJobSpec to create a job.
 * To create a job:
 * POST    /api/v1/jobs
 *
 * To list the jobs
 * GET     /api/v1/jobs
 *
 * To get a specific job
 * GET     /api/v1/jobs/{id}
 *
 * To delete a job by id
 * DELETE  /api/v1/jobs/{id}
 * */
@Path(RestConstants.V1 + "/" + RestConstants.JOBS)
@Produces({MediaType.APPLICATION_JSON + "; " + RestConstants.CHARSET_UTF8})
public class JobApi {

  // A ping test to verify the job server is up.
  @Path(RestConstants.PING)
  @GET
  @Consumes(MediaType.APPLICATION_JSON)
  public Response ping() {
    return new JsonResponse.Builder<String>(Response.Status.OK)
        .success(true).result("Pong").build();
  }

  @POST
  @Consumes({RestConstants.MEDIA_TYPE_YAML, MediaType.APPLICATION_JSON})
  public Response submitJob(MLJobSpec jobSpec) {
    // Submit the job spec through submitter
    return new JsonResponse.Builder<MLJobSpec>(Response.Status.ACCEPTED)
        .success(true).result(jobSpec).build();
  }

  @GET
  @Path("{" + RestConstants.JOB_ID + "}")
  public Response listJob(@PathParam(RestConstants.JOB_ID) String id) {
    // Query the job status though submitter
    return new JsonResponse.Builder<MLJobSpec>(Response.Status.OK)
        .success(true).result(id).build();
  }

  @GET
  public Response listAllJob() {
    // Query all the job status though submitter
    return new JsonResponse.Builder<MLJobSpec>(Response.Status.OK)
        .success(true).build();
  }

  @DELETE
  @Path("{" + RestConstants.JOB_ID + "}")
  public Response deleteJob(@PathParam(RestConstants.JOB_ID) String id) {
    // Delete the job though submitter
    return new JsonResponse.Builder<MLJobSpec>(Response.Status.OK)
        .success(true).result(id).build();
  }

}
