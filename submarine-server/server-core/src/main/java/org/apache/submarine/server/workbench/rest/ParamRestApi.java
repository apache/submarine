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
package org.apache.submarine.server.workbench.rest;

import com.google.gson.Gson;
import org.apache.submarine.server.workbench.annotation.SubmarineApi;
import org.apache.submarine.server.workbench.database.entity.Param;
import org.apache.submarine.server.workbench.database.service.ParamService;
import org.apache.submarine.server.response.JsonResponse;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

import javax.inject.Inject;
import javax.inject.Singleton;
import javax.ws.rs.DELETE;
import javax.ws.rs.GET;
import javax.ws.rs.POST;
import javax.ws.rs.PUT;
import javax.ws.rs.Path;
import javax.ws.rs.Produces;
import javax.ws.rs.QueryParam;
import javax.ws.rs.core.Response;

@Path("/param")
@Produces("application/json")
@Singleton
public class ParamRestApi {
  private static final Logger LOG = LoggerFactory.getLogger(LoginRestApi.class);
  private static final Gson gson = new Gson();
  ParamService paramService = new ParamService();

  @Inject
  public ParamRestApi() {
  }

  @GET
  @Path("/list")
  @SubmarineApi
  public Response listParam(@QueryParam("id") String id) {
    LOG.info("getParam ({})", id);
    
    List<Param> params;
    try {
      params = paramService.selectAll();

    } catch (Exception e) {

      LOG.error(e.toString());
      e.printStackTrace();
      return new JsonResponse.Builder<String>(Response.Status.OK).success(true).result("error").build();
    }
    return new JsonResponse.Builder<List<Param>>(Response.Status.OK).success(true).result(params).build();
  }

  @GET
  @Path("/")
  @SubmarineApi
  public Response getParam(@QueryParam("id") String id) {
    LOG.info("getParam ({})", id);
    
    Param param;
    try {
      param = paramService.selectById(id);

    } catch (Exception e) {

      LOG.error(e.toString());
      e.printStackTrace();
      return new JsonResponse.Builder<String>(Response.Status.OK).success(true).result("error").build();
    }
    return new JsonResponse.Builder<Param>(Response.Status.OK).success(true).result(param).build();
  }

  @POST
  @Path("/")
  @SubmarineApi
  public Response postParam(Param param) {
    LOG.info("postParam ({})", param);
    boolean result = false;
    try {
      result = paramService.insert(param);
    } catch (Exception e) {

      LOG.error(e.toString());
      e.printStackTrace();
      return new JsonResponse.Builder<String>(Response.Status.OK).success(true).result("error").build();
    }
    return new JsonResponse.Builder<Boolean>(Response.Status.OK).success(true).result(result).build();
  }

  @DELETE
  @Path("/")
  @SubmarineApi
  public Response deleteParam(@QueryParam("id") String id) {
    LOG.info("deleteParam ({})", id);
    boolean result = false;
    try {
      result = paramService.deleteById(id);
    } catch (Exception e) {
      LOG.error(e.toString());
      e.printStackTrace();
      return new JsonResponse.Builder<String>(Response.Status.OK).success(true).result("error").build();
    }
    return new JsonResponse.Builder<Boolean>(Response.Status.OK).success(true).result(result).build();
  }

  @PUT
  @Path("")
  @SubmarineApi
  public Response putParam(Param param) {
    LOG.info("putParam ({})", param);
    boolean result = false;
    try {
      result = paramService.update(param);
    } catch (Exception e) {
      LOG.error(e.toString());
      e.printStackTrace();
      return new JsonResponse.Builder<String>(Response.Status.OK).success(true).result("error").build();
    }
    return new JsonResponse.Builder<Boolean>(Response.Status.OK).success(true).result(result).build();
  }

  @POST
  @Path("/selective")
  @SubmarineApi
  public Response selectByPrimaryKeySelective(Param metric) {
    List<Param> params;
    try {
      params = paramService.selectByPrimaryKeySelective(metric);

    } catch (Exception e) {

      LOG.error(e.toString());
      e.printStackTrace();
      return new JsonResponse.Builder<String>(Response.Status.OK).success(true).result("error").build();
    }
    return new JsonResponse.Builder<List<Param>>(Response.Status.OK).success(true).result(params).build();
  }

}
