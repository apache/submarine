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
package org.apache.submarine.server.rest.workbench;

import org.apache.submarine.server.utils.response.JsonResponse;
import org.apache.submarine.server.rest.workbench.annotation.SubmarineApi;
import org.apache.submarine.server.database.workbench.entity.ParamEntity;
import org.apache.submarine.server.database.workbench.service.ParamService;
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
import javax.ws.rs.PathParam;
import javax.ws.rs.Produces;
import javax.ws.rs.QueryParam;
import javax.ws.rs.core.Response;

@Path("/param")
@Produces("application/json")
@Singleton
public class ParamRestApi {
  private static final Logger LOG = LoggerFactory.getLogger(LoginRestApi.class);
  ParamService paramService = new ParamService();

  @Inject
  public ParamRestApi() {
  }

  @GET
  @Path("/list")
  @SubmarineApi
  public Response listParam(@QueryParam("id") String id,
                            @QueryParam("paramKey") String paramKey,
                            @QueryParam("value") String value,
                            @QueryParam("workerIndex") String workerIndex) {

    ParamEntity param = new ParamEntity();
    param.setId(id);
    param.setKey(paramKey);
    param.setValue(value);
    param.setWorkerIndex(workerIndex);

    LOG.info("listParam ({})", param);

    List<ParamEntity> params;
    try {
      params = paramService.selectByPrimaryKeySelective(param);
    } catch (Exception e) {
      LOG.error(e.toString());
      return new JsonResponse.Builder<Boolean>(Response.Status.OK).success(false).build();
    }
    return new JsonResponse.Builder<List<ParamEntity>>(Response.Status.OK).success(true).
            result(params).build();
  }

  @GET
  @Path("/{id}")
  @SubmarineApi
  public Response getParam(@PathParam("id") String id) {
    LOG.info("getParam ({})", id);

    ParamEntity param;
    try {
      param = paramService.selectById(id);
    } catch (Exception e) {
      LOG.error(e.toString());
      return new JsonResponse.Builder<Boolean>(Response.Status.OK).success(false).build();
    }
    return new JsonResponse.Builder<ParamEntity>(Response.Status.OK).success(true).result(param).build();
  }

  @POST
  @Path("/add")
  @SubmarineApi
  public Response postParam(ParamEntity param) {
    LOG.info("postParam ({})", param);
    boolean result;
    try {
      result = paramService.insert(param);
    } catch (Exception e) {
      LOG.error(e.toString());
      return new JsonResponse.Builder<Boolean>(Response.Status.OK).success(false).build();
    }
    return new JsonResponse.Builder<Boolean>(Response.Status.OK).success(true).result(result).build();
  }

  @DELETE
  @Path("/delete")
  @SubmarineApi
  public Response deleteParam(@QueryParam("id") String id) {
    LOG.info("deleteParam ({})", id);
    boolean result;
    try {
      result = paramService.deleteById(id);
    } catch (Exception e) {
      LOG.error(e.toString());
      return new JsonResponse.Builder<Boolean>(Response.Status.OK).success(false).build();
    }
    return new JsonResponse.Builder<Boolean>(Response.Status.OK).success(true).result(result).build();
  }

  @PUT
  @Path("/edit")
  @SubmarineApi
  public Response putParam(ParamEntity param) {
    LOG.info("putParam ({})", param);
    boolean result = false;
    try {
      result = paramService.update(param);
    } catch (Exception e) {
      LOG.error(e.toString());
      return new JsonResponse.Builder<Boolean>(Response.Status.OK).success(false).build();
    }
    return new JsonResponse.Builder<Boolean>(Response.Status.OK).success(true).result(result).build();
  }

  @POST
  @Path("/selective")
  @SubmarineApi
  public Response selectByPrimaryKeySelective(ParamEntity metric) {
    List<ParamEntity> params;
    try {
      params = paramService.selectByPrimaryKeySelective(metric);
    } catch (Exception e) {
      LOG.error(e.toString());
      return new JsonResponse.Builder<Boolean>(Response.Status.OK).success(false).build();
    }
    return new JsonResponse.Builder<List<ParamEntity>>(Response.Status.OK).
            success(true).result(params).build();
  }
}
