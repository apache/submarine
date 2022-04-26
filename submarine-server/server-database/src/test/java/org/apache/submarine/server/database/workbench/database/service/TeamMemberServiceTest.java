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
package org.apache.submarine.server.database.workbench.service;

import org.apache.submarine.server.database.workbench.entity.TeamMemberEntity;
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
    List<TeamMemberEntity> teamMemberList = teamMemberService.queryList("teamId");
    assertTrue(teamMemberList.size() > 0);
    for (TeamMemberEntity member : teamMemberList) {
      teamMemberService.deleteByPrimaryKey(member.getId());
    }
  }

  @Test
  public void insertSelective() throws Exception {
    TeamMemberEntity teamMember = new TeamMemberEntity();
    teamMember.setTeamId("teamId");
    teamMember.setTeamName("teamName");
    teamMember.setCreateBy("createBy");
    teamMember.setMember("member");
    teamMember.setInviter(1);
    Boolean ret = teamMemberService.insertSelective(teamMember);
    assertTrue(ret);

    List<TeamMemberEntity> teamMemberList = teamMemberService.queryList("teamId");
    assertEquals(teamMemberList.size(), 1);
    for (TeamMemberEntity member : teamMemberList) {
      assertEquals(member.getTeamName(), teamMember.getTeamName());
      assertEquals(member.getTeamId(), teamMember.getTeamId());
      assertEquals(member.getCreateBy(), teamMember.getCreateBy());
      assertEquals(member.getMember(), teamMember.getMember());
      assertEquals(member.getInviter(), teamMember.getInviter());

      LOG.info("member.createTime:{}, member.updateTime:{}", member.getCreateTime(), member.getUpdateTime());
    }
  }
}
