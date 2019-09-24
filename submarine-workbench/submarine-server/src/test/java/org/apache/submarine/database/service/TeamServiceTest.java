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
package org.apache.submarine.database.service;

import org.apache.submarine.database.entity.Team;
import org.apache.submarine.database.entity.TeamMember;
import org.junit.After;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.util.ArrayList;
import java.util.List;
import static junit.framework.TestCase.assertEquals;
import static org.junit.Assert.assertTrue;

public class TeamServiceTest {

  private static final Logger LOG = LoggerFactory.getLogger(TeamServiceTest.class);
  TeamService teamService = new TeamService();

  @After
  public void removeAllRecord() throws Exception {
    List<Team> teamList = teamService.queryPageList(null, "create_time", "desc", 0, 100);
    LOG.info("teamList.size():{}", teamList.size());
    for (Team team : teamList) {
      teamService.delete(team.getId());
    }
  }

  @Test
  public void queryPageList() throws Exception {
    TeamMember teamMember = new TeamMember();
    teamMember.setTeamName("submarine");
    teamMember.setInviter(0);
    teamMember.setMember("admin");
    teamMember.setCreateBy("createByteamMember");

    Team team = new Team();
    team.setTeamName("submarine");
    team.setOwner("test_sub");
    team.setCreateBy("createByteam");
    List list = new ArrayList<TeamMember>();
    list.add(teamMember);
    team.setCollaborators(list);
    Boolean ret = teamService.add(team);
    assertTrue(ret);

    List<Team> teamList = teamService.queryPageList("test_sub", "create_time", "desc", 0, 100);
    assertEquals(teamList.size(), 1);
    Team team_db = teamList.get(0);
    assertEquals(team.getTeamName(), team_db.getTeamName());
    assertEquals(team.getOwner(), team_db.getOwner());
    assertEquals(team.getCreateBy(), team_db.getCreateBy());

    assertEquals(team_db.getCollaborators().size(), 1);
    TeamMember teamMember_db = team_db.getCollaborators().get(0);
    assertEquals(team.getId(), teamMember_db.getTeamId());
    assertEquals(teamMember.getTeamName(), teamMember_db.getTeamName());
    assertEquals(teamMember.getInviter(), teamMember_db.getInviter());
    assertEquals(teamMember.getMember(), teamMember_db.getMember());
    assertEquals(teamMember.getCreateBy(), teamMember_db.getCreateBy());
  }

  @Test
  public void updateByPrimaryKeySelective() throws Exception {
    TeamMember teamMember = new TeamMember();
    teamMember.setTeamName("submarine");
    teamMember.setInviter(0);
    teamMember.setMember("admin");
    teamMember.setCreateBy("createByteamMember");

    Team team = new Team();
    team.setTeamName("submarine");
    team.setOwner("test_sub");
    team.setCreateBy("createByteam");
    List list = new ArrayList<TeamMember>();
    list.add(teamMember);
    team.setCollaborators(list);
    Boolean ret = teamService.add(team);
    assertTrue(ret);

    team.setTeamName("submarine_update");
    TeamMember teamMember_update = new TeamMember();
    teamMember_update.setTeamName("submarine");
    teamMember_update.setInviter(0);
    teamMember_update.setMember("test_member");
    teamMember_update.setCreateBy("createByteamMember2");
    list.add(teamMember_update);

    boolean editRet = teamService.updateByPrimaryKeySelective(team);
    assertTrue(editRet);
    List<Team> teamList = teamService.queryPageList("test_sub", "create_time", "desc", 0, 100);
    assertEquals(teamList.size(), 1);

    Team team_db = teamList.get(0);
    assertEquals(team.getTeamName(), team_db.getTeamName());
    List<TeamMember> teamMemberList = team_db.getCollaborators();
    assertEquals(teamMemberList.size(), 2);
    for (TeamMember member : teamMemberList) {
      assertEquals(member.getTeamName(), team.getTeamName());
    }

  }

  @Test
  public void delete() throws Exception {
    TeamMember teamMember = new TeamMember();
    teamMember.setTeamName("submarine");
    teamMember.setInviter(0);
    teamMember.setMember("admin");
    teamMember.setCreateBy("createByteamMember");

    Team team = new Team();
    team.setTeamName("submarine");
    team.setOwner("test_sub");
    team.setCreateBy("createByteam");
    List list = new ArrayList<TeamMember>();
    list.add(teamMember);
    team.setCollaborators(list);
    Boolean ret = teamService.add(team);
    assertTrue(ret);

    Boolean deleteRet = teamService.delete(team.getId());
    assertTrue(deleteRet);

    List<Team> teamList = teamService.queryPageList("test_sub", "create_time", "desc", 0, 100);
    assertEquals(teamList.size(), 0);

    TeamMemberService teamMemberService = new TeamMemberService();
    List<TeamMember> teamMemberList = teamMemberService.queryList(team.getTeamName());
    assertEquals(teamMemberList.size(), 0);
  }
}
