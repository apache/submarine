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

import org.apache.submarine.server.response.JsonResponse;
import org.apache.submarine.server.rest.workbench.annotation.SubmarineApi;
import org.apache.submarine.server.workbench.database.entity.MetricEntity;
import org.apache.submarine.server.workbench.database.service.MetricService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

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
import java.sql.Timestamp;
import java.util.List;

@Path("/metric")
@Produces("application/json")
@Singleton
public class MetricRestApi {
  private static final Logger LOG = LoggerFactory.getLogger(LoginRestApi.class);
  MetricService metricService = new MetricService();

  @Inject
  public MetricRestApi() {
  }

  @GET
  @Path("/list")
  @SubmarineApi
  public Response listMetric(@QueryParam("metricKey") String metricKey,
                              @QueryParam("value") Float value,
                              @QueryParam("workerIndex") String workerIndex,
                              @QueryParam("timestamp") Timestamp timestamp,
                              @QueryParam("step") Integer step,
                              @QueryParam("isNan") Boolean isNan,
                              @QueryParam("id") String id) {

    MetricEntity metric = new MetricEntity();
    metric.setKey(metricKey);
    metric.setValue(value);
    metric.setWorkerIndex(workerIndex);
    metric.setTimestamp(timestamp);
    metric.setStep(step);
    metric.setIsNan(isNan);
    metric.setId(id);

    LOG.info("listMetric ({})", metric);

    List<MetricEntity> metrics;
    try {
      metrics = metricService.selectByPrimaryKeySelective(metric);
    } catch (Exception e) {
      LOG.error(e.toString());
      return new JsonResponse.Builder<Boolean>(Response.Status.OK).success(false).build();
    }
    return new JsonResponse.Builder<List<MetricEntity>>(Response.Status.OK).success(true).
            result(metrics).build();
  }

  @GET
  @Path("/{id}")
  @SubmarineApi
  public Response getMetric(@PathParam("id") String id) {
    MetricEntity metric;
    try {
      metric = metricService.selectById(id);
    } catch (Exception e) {
      LOG.error(e.toString());
      return new JsonResponse.Builder<Boolean>(Response.Status.OK).success(true).build();
    }
    return new JsonResponse.Builder<MetricEntity>(Response.Status.OK).success(true).result(metric).build();
  }

  @POST
  @Path("/add")
  @SubmarineApi
  public Response postMetric(MetricEntity metric) {
    boolean result = false;
    try {
      result = metricService.insert(metric);
    } catch (Exception e) {
      LOG.error(e.toString());
      return new JsonResponse.Builder<Boolean>(Response.Status.OK).success(false).build();
    }
    return new JsonResponse.Builder<Boolean>(Response.Status.OK).success(true).result(result).build();
  }

  @DELETE
  @Path("/delete")
  @SubmarineApi
  public Response deleteMetric(@QueryParam("id") String id) {
    boolean result = false;
    try {
      result = metricService.deleteById(id);
    } catch (Exception e) {
      LOG.error(e.toString());
      return new JsonResponse.Builder<Boolean>(Response.Status.OK).success(false).build();
    }
    return new JsonResponse.Builder<Boolean>(Response.Status.OK).success(true).result(result).build();
  }

  @PUT
  @Path("/edit")
  @SubmarineApi
  public Response putMetric(MetricEntity metric) {
    boolean result = false;
    try {
      result = metricService.update(metric);
    } catch (Exception e) {
      LOG.error(e.toString());
      return new JsonResponse.Builder<Boolean>(Response.Status.OK).success(false).build();
    }
    return new JsonResponse.Builder<Boolean>(Response.Status.OK).success(true).result(result).build();
  }

  @POST
  @Path("/selective")
  @SubmarineApi
  public Response selectByPrimaryKeySelective(MetricEntity metric) {
    List<MetricEntity> metrics;
    try {
      metrics = metricService.selectByPrimaryKeySelective(metric);
    } catch (Exception e) {
      LOG.error(e.toString());
      return new JsonResponse.Builder<Boolean>(Response.Status.OK).success(false).build();
    }
    return new JsonResponse.Builder<List<MetricEntity>>(Response.Status.OK).success(true).
            result(metrics).build();
  }
}
