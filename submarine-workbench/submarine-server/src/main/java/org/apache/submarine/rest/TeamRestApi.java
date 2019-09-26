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
import org.apache.submarine.database.entity.Team;
import org.apache.submarine.database.service.SysMessageService;
import org.apache.submarine.database.service.TeamService;
import org.apache.submarine.server.JsonResponse;
import org.apache.submarine.server.JsonResponse.ListResult;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.inject.Inject;
import javax.inject.Singleton;
import javax.ws.rs.POST;
import javax.ws.rs.Path;
import javax.ws.rs.GET;
import javax.ws.rs.Produces;
import javax.ws.rs.DELETE;
import javax.ws.rs.PUT;
import javax.ws.rs.QueryParam;
import javax.ws.rs.core.Response;
import java.util.ArrayList;
import java.util.List;

@Path("/team")
@Produces("application/json")
@Singleton
public class TeamRestApi {
  private static final Logger LOG = LoggerFactory.getLogger(TeamRestApi.class);

  private TeamService teamService = new TeamService();

  @Inject
  public TeamRestApi() {
  }

  @GET
  @Path("/list")
  @SubmarineApi
  public Response list(@QueryParam("owner") String owner,
                       @QueryParam("column") String column,
                       @QueryParam("order") String order,
                       @QueryParam("pageNo") int pageNo,
                       @QueryParam("pageSize") int pageSize) {
    LOG.info("TeamRestApi.list() owner:{}, pageNo:{}, pageSize:{}", owner, pageNo, pageSize);

    List<Team> teams = new ArrayList<>();
    try {
      // TODO(zhulinhao): Front need to correct 'owner' value, and Whether need the
      //  front to create_by value（At the time of pr commited）
      teams = teamService.queryPageList(owner, column, order, pageNo, pageSize);
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      return new JsonResponse.Builder<>(Response.Status.OK).success(false).build();
    }
    PageInfo<Team> page = new PageInfo<>(teams);
    ListResult<Team> listResult = new ListResult(teams, page.getTotal());
    return new JsonResponse.Builder<ListResult<Team>>(Response.Status.OK)
        .success(true).result(listResult).build();
  }

  @POST
  @Path("/add")
  @SubmarineApi
  public Response add(Team team) {
    LOG.info("add team:{}", team.toString());

    // insert into database, return id
    try {
      teamService.add(team);
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      return new JsonResponse.Builder<>(Response.Status.OK).success(false)
          .message("Save team failed!").build();
    }


    // TODO(zhulinhao): add message
    // For each of the members, increase the invitation information saved to sys_message table
    /**SysMessage sysMessage = new SysMessage();
     try {
     sysMessageService.add(sysMessage);
     } catch (Exception e) {
     LOG.error(e.getMessage(), e);
     return new JsonResponse.Builder<>(Response.Status.OK).success(false)
     .message("Save team failed!").build();
     }*/

    return new JsonResponse.Builder<Team>(Response.Status.OK)
        .message("Save team successfully!").result(team).success(true).build();
  }

  @PUT
  @Path("/edit")
  @SubmarineApi
  public Response edit(Team team) {
    LOG.info("edit team:{}", team.toString());

    // TODO(zhulinhao): need set update_by value
    try {
      // update team
      teamService.updateByPrimaryKeySelective(team);

      // TODO(zhulinhao)
      // Save inviter=0 in the newly added member and the invitation
      // message to join the team that has not been sent into the message
      // table sys_message to avoid sending the invitation message repeatedly

    } catch (Exception e) {
      return new JsonResponse.Builder<>(Response.Status.OK).success(false)
          .message("update team failed!").build();
    }

    return new JsonResponse.Builder<>(Response.Status.OK)
        .message("update team successfully!").success(true).build();
  }

  @DELETE
  @Path("/delete")
  @SubmarineApi
  public Response delete(@QueryParam("id") String id) {
    // TODO(zhulinhao): At the front desk need to id
    LOG.info("delete team:{}", id);

    // Delete data in a team and team_member table
    // TODO(zhulinhao):delete sys_message's invite messages
    try {
      teamService.delete(id);
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      return new JsonResponse.Builder<>(Response.Status.OK).success(false)
          .message("delete team failed!").build();
    }

    return new JsonResponse.Builder<>(Response.Status.OK)
        .message("Delete team successfully!").success(true).build();
  }
}
