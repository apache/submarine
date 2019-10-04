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
package org.apache.submarine.rest;

import com.github.pagehelper.PageInfo;
import org.apache.submarine.annotation.SubmarineApi;
import org.apache.submarine.database.entity.Project;
import org.apache.submarine.database.service.ProjectService;
import org.apache.submarine.server.JsonResponse;
import org.apache.submarine.server.JsonResponse.ListResult;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

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
import java.util.ArrayList;
import java.util.List;

@Path("/project")
@Produces("application/json")
@Singleton
public class ProjectRestApi {
  private static final Logger LOG = LoggerFactory.getLogger(ProjectRestApi.class);

  private ProjectService projectService = new ProjectService();

  @Inject
  public ProjectRestApi() {
  }

  @GET
  @Path("/list")
  @SubmarineApi
  public Response list(@QueryParam("userName") String userName,
                       @QueryParam("column") String column,
                       @QueryParam("order") String order,
                       @QueryParam("pageNo") int pageNo,
                       @QueryParam("pageSize") int pageSize) {
    LOG.info("ProjectRestApi.list() owner:{}, pageNo:{}, pageSize:{}", userName, pageNo, pageSize);

    List<Project> projectList = new ArrayList<>();
    try {
      projectList = projectService.queryPageList(userName, column, order, pageNo, pageSize);
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      return new JsonResponse.Builder<>(Response.Status.OK).success(false).build();
    }
    PageInfo<Project> page = new PageInfo<>(projectList);
    ListResult<Project> listResult = new ListResult(projectList, page.getTotal());
    return new JsonResponse.Builder<ListResult<Project>>(Response.Status.OK)
        .success(true).result(listResult).build();
  }

  @POST
  @Path("/add")
  @SubmarineApi
  public Response add(Project project) {
    LOG.info("add project:{}", project.toString());

    // insert into database, return id
    try {
      projectService.add(project);
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      return new JsonResponse.Builder<>(Response.Status.OK).success(false)
          .message("Save project failed!").build();
    }

    return new JsonResponse.Builder<Project>(Response.Status.OK)
        .message("Save project successfully!").result(project).success(true).build();
  }

  @PUT
  @Path("/edit")
  @SubmarineApi
  public Response edit(Project project) {
    LOG.info("edit project:{}", project.toString());

    try {
      // update project
      projectService.updateByPrimaryKeySelective(project);
    } catch (Exception e) {
      return new JsonResponse.Builder<>(Response.Status.OK).success(false)
          .message("Update project failed!").build();
    }

    return new JsonResponse.Builder<>(Response.Status.OK)
        .message("Update project successfully!").success(true).build();
  }

  @DELETE
  @Path("/delete")
  @SubmarineApi
  public Response delete(@QueryParam("id") String id) {
    // TODO(zhulinhao): At the front desk need to id
    LOG.info("delete project:{}", id);

    try {
      projectService.delete(id);
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      return new JsonResponse.Builder<>(Response.Status.OK).success(false)
          .message("Delete project failed!").build();
    }

    return new JsonResponse.Builder<>(Response.Status.OK)
        .message("Delete project successfully!").success(true).build();
  }
}
