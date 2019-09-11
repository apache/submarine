/**
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * <p>
 * http://www.apache.org/licenses/LICENSE-2.0
 * <p>
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. See accompanying LICENSE file.
 */
package org.apache.submarine.rest;

import com.github.pagehelper.PageInfo;
import com.google.gson.Gson;
import org.apache.submarine.annotation.SubmarineApi;
import org.apache.submarine.database.entity.QueryResult;
import org.apache.submarine.database.entity.Team;
import org.apache.submarine.database.entity.TeamMemeber;
import org.apache.submarine.server.JsonResponse;
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
import java.util.Arrays;
import java.util.Date;
import java.util.List;

@Path("/team")
@Produces("application/json")
@Singleton
public class TeamRestApi {
  private static final Logger LOG = LoggerFactory.getLogger(TeamRestApi.class);

  private static final Gson gson = new Gson();

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

    // mock data
    List<Team> teams = new ArrayList<>();
    for (int i = 0; i < 3; i++) {
      Team team = new Team();
      team.setId("team" + i);

      // test different owner
      if (i == 0) {
        team.setOwner("admin");
      } else if (i == 1) {
        team.setOwner("liuxun");
      } else {
        team.setOwner("test");
      }

      team.setTeamName("test-team" + i);

      for (int j = 0; j < 3; j++) {
        TeamMemeber memeber = new TeamMemeber();
        memeber.setId("team" + i + "name" + j + "-id");
        memeber.setMember("team" + i + "name" + j);
        if (j == 0) {
          memeber.setInviter(0);
        } else {
          memeber.setInviter(1);
        }
        team.addCollaborator(memeber);
      }
      teams.add(team);
    }

    PageInfo<Team> page = new PageInfo<>(teams);
    QueryResult<Team> queryResult = new QueryResult(teams, page.getTotal());
    return new JsonResponse.Builder<QueryResult>(Response.Status.OK)
        .success(true).result(queryResult).build();
  }

  @POST
  @Path("/add")
  @SubmarineApi
  public Response add(Team team) {
    LOG.info("add team:{}", team.toString());

    // insert into database, return id
    Date mockId = new Date();
    team.setId(String.valueOf(mockId.getTime()));

    // 为每个成员，增加邀请信息保存到 sys_message 表中

    QueryResult<Team> queryResult = new QueryResult(Arrays.asList(team), 1);
    return new JsonResponse.Builder<>(Response.Status.OK)
        .message("Save team successfully!").result(queryResult).success(true).build();
  }

  @PUT
  @Path("/edit")
  @SubmarineApi
  public Response edit(Team team) {
    LOG.info("edit team:{}", team.toString());

    // todo(zhulinhao)
    // 把 team 保存到 team 表中

    List<TeamMemeber> memebers = team.getCollaborators();
    // 将每个 TeamMemeber 保存到 team_member table 中
    // 前端有如下情况：
    // 1. 前端没有对成员进行修改
    // 2. 前端新增了成员
    // 3. 前端删除了成员

    // 将新增的成员中inviter=0，并且没有发送过加入 team 邀请信息
    // 保存到消息表 sys_message 中，避免重复发送邀请信息

    return new JsonResponse.Builder<>(Response.Status.OK)
        .message("Save team successfully!").success(true).build();
  }

  @DELETE
  @Path("/delete")
  @SubmarineApi
  public Response delete(@QueryParam("id") String id) {
    LOG.info("delete team:{}", id);

    // 删除 team 和 team_member 表中的数据
    // 删除 sys_message 中的邀请信息

    return new JsonResponse.Builder<>(Response.Status.OK)
        .message("Delete team successfully!").success(true).build();
  }
}
