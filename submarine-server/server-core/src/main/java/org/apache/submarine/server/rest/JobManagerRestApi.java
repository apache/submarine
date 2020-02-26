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

import org.apache.submarine.server.JobManager;
import org.apache.submarine.server.api.exception.UnsupportedJobTypeException;
import org.apache.submarine.server.api.job.Job;
import org.apache.submarine.server.api.job.JobId;
import org.apache.submarine.server.api.spec.JobSpec;
import org.apache.submarine.server.response.JsonResponse;

import javax.ws.rs.Consumes;
import javax.ws.rs.DELETE;
import javax.ws.rs.GET;
import javax.ws.rs.POST;
import javax.ws.rs.Path;
import javax.ws.rs.PathParam;
import javax.ws.rs.Produces;
import javax.ws.rs.core.MediaType;
import javax.ws.rs.core.Response;
import java.util.ArrayList;
import java.util.List;

/**
 * {@link JobManager}'s REST API v1. It can accept {@link JobSpec} to create a job.
 */
@Path(RestConstants.V1 + "/" + RestConstants.JOBS)
@Produces({MediaType.APPLICATION_JSON + "; " + RestConstants.CHARSET_UTF8})
public class JobManagerRestApi {

  /**
   * Return the Pong message for test the connectivity
   * @return Pong message
   */
  @GET
  @Path(RestConstants.PING)
  @Consumes(MediaType.APPLICATION_JSON)
  public Response ping() {
    return new JsonResponse.Builder<String>(Response.Status.OK)
        .success(true).result("Pong").build();
  }

  /**
   * Returns the contents of {@link Job} that submitted by user.
   * @param jobSpec job spec
   * @return the contents of job
   */
  @POST
  @Consumes({RestConstants.MEDIA_TYPE_YAML, MediaType.APPLICATION_JSON})
  public Response submitJob(JobSpec jobSpec) {
    if (!jobSpec.validate()) {
      return new JsonResponse.Builder<String>(Response.Status.ACCEPTED)
          .success(false).result("Invalid params.").build();
    }

    try {
      Job job = JobManager.getInstance().submitJob(jobSpec);
      return new JsonResponse.Builder<Job>(Response.Status.OK)
          .success(true).result(job).build();
    } catch (UnsupportedJobTypeException e) {
      return new JsonResponse.Builder<String>(Response.Status.ACCEPTED)
          .success(false).result(e.getMessage()).build();
    }
  }

  /**
   * List all job for the user
   * @return job list
   */
  @GET
  public Response listJob() {
    // TODO(jiwq): Hook JobManager when 0.4.0 released
    return new JsonResponse.Builder<List<Job>>(Response.Status.OK)
        .success(true).result(new ArrayList<>()).build();
  }

  /**
   * Returns the job detailed info by specified job id
   * @param id job id
   * @return the detailed info of job
   */
  @GET
  @Path("{" + RestConstants.JOB_ID + "}")
  public Response getJob(@PathParam(RestConstants.JOB_ID) String id) {
    // TODO(jiwq): Hook JobManager when 0.4.0 released
    Job job = new Job();
    job.setJobId(JobId.fromString(id));
    return new JsonResponse.Builder<Job>(Response.Status.OK)
        .success(true).result(job).build();
  }

  /**
   * Returns the job that deleted
   * @param id job id
   * @return the detailed info about deleted job
   */
  @DELETE
  @Path("{" + RestConstants.JOB_ID + "}")
  public Response deleteJob(@PathParam(RestConstants.JOB_ID) String id) {
    // TODO(jiwq): Hook JobManager when 0.4.0 released
    return new JsonResponse.Builder<Job>(Response.Status.OK)
        .success(true).result(new Job()).build();
  }
}
