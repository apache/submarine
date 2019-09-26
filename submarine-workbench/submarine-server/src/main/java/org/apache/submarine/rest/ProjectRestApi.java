/**
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. See accompanying LICENSE file.
 */
package org.apache.submarine.rest;

import com.github.pagehelper.PageInfo;
import org.apache.submarine.annotation.SubmarineApi;
import org.apache.submarine.database.entity.Project;
import org.apache.submarine.database.entity.Team;
import org.apache.submarine.database.service.ProjectService;
import org.apache.submarine.database.service.TeamService;
import org.apache.submarine.server.JsonResponse;
import org.apache.submarine.server.JsonResponse.ListResult;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.inject.Inject;
import javax.inject.Singleton;
import javax.ws.rs.GET;
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
      projectList = projectService.queryPageList("liuxun", column, order, pageNo, pageSize);
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      return new JsonResponse.Builder<>(Response.Status.OK).success(false).build();
    }
    PageInfo<Project> page = new PageInfo<>(projectList);
    ListResult<Project> listResult = new ListResult(projectList, page.getTotal());
    return new JsonResponse.Builder<ListResult<Project>>(Response.Status.OK)
        .success(true).result(listResult).build();
  }

}
