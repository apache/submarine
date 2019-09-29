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

import org.apache.submarine.database.entity.TeamMember;
import org.junit.After;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class TeamMemberServiceTest {
  private static final Logger LOG = LoggerFactory.getLogger(TeamMemberServiceTest.class);
  TeamMemberService teamMemberService = new TeamMemberService();

  @After
  public void removeAllRecord() throws Exception {
    List<TeamMember> teamMemberList = teamMemberService.queryList("teamId");
    assertTrue(teamMemberList.size() > 0);
    for (TeamMember member : teamMemberList) {
      teamMemberService.deleteByPrimaryKey(member.getId());
    }
  }

  @Test
  public void insertSelective() throws Exception {
    TeamMember teamMember = new TeamMember();
    teamMember.setTeamId("teamId");
    teamMember.setTeamName("teamName");
    teamMember.setCreateBy("createBy");
    teamMember.setMember("member");
    teamMember.setInviter(1);
    Boolean ret = teamMemberService.insertSelective(teamMember);
    assertTrue(ret);

    List<TeamMember> teamMemberList = teamMemberService.queryList("teamId");
    assertEquals(teamMemberList.size(), 1);
    for (TeamMember member : teamMemberList) {
      assertEquals(member.getTeamName(), teamMember.getTeamName());
      assertEquals(member.getTeamId(), teamMember.getTeamId());
      assertEquals(member.getCreateBy(), teamMember.getCreateBy());
      assertEquals(member.getMember(), teamMember.getMember());
      assertEquals(member.getInviter(), teamMember.getInviter());

      LOG.info("member.createTime:{}, member.updateTime:{}", member.getCreateTime(), member.getUpdateTime());
    }
  }
}