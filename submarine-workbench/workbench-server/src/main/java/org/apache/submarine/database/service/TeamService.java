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
package org.apache.submarine.database.service;

import org.apache.ibatis.session.RowBounds;
import org.apache.ibatis.session.SqlSession;
import org.apache.submarine.database.MyBatisUtil;
import org.apache.submarine.database.entity.Team;
import org.apache.submarine.database.entity.TeamMember;
import org.apache.submarine.database.mappers.TeamMapper;
import org.apache.submarine.database.mappers.TeamMemberMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class TeamService {
  private static final Logger LOG = LoggerFactory.getLogger(TeamService.class);

  public List<Team> queryPageList(String owner,
                                  String column,
                                  String order,
                                  int pageNo,
                                  int pageSize) throws Exception {
    LOG.info("queryPageList owner:{}, column:{}, order:{}, pageNo:{}, pageSize:{}",
        owner, column, order, pageNo, pageSize);

    List<Team> list = null;
    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      TeamMapper teamMapper = sqlSession.getMapper(TeamMapper.class);
      Map<String, Object> where = new HashMap<>();
      where.put("owner", owner);
      where.put("column", column);
      where.put("order", order);
      list = teamMapper.selectAll(where, new RowBounds(pageNo, pageSize));

      TeamMemberMapper teamMemberMapper = sqlSession.getMapper(TeamMemberMapper.class);
      // query from team_member table, and set to team
      for (Team team : list) {
        Map<String, Object> whereMember = new HashMap<>();
        whereMember.put("teamId", team.getId());
        List<TeamMember> teamMembers = teamMemberMapper.selectAll(whereMember);
        for (TeamMember teamMember : teamMembers) {
          team.addCollaborator(teamMember);
        }
      }
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new Exception(e);
    }
    return list;
  }

  public boolean add(Team team) throws Exception {
    LOG.info("add({})", team.toString());

    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      TeamMapper teamMapper = sqlSession.getMapper(TeamMapper.class);
      teamMapper.insert(team);

      TeamMemberMapper teamMemberMapper = sqlSession.getMapper(TeamMemberMapper.class);
      // add teamMember, when add team, should insert 'Collaborators' to team_member
      List<TeamMember> list = team.getCollaborators();
      for (TeamMember teamMember : list) {
        // TODO(zhulinhao): teamMember's member is sys_user's id now.
        teamMember.setTeamId(team.getId());
        teamMemberMapper.insert(teamMember);
      }

      sqlSession.commit();
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new Exception(e);
    }
    return true;
  }

  public boolean updateByPrimaryKeySelective(Team team) throws Exception {
    LOG.info("updateByPrimaryKeySelective({})", team.toString());

    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      TeamMapper teamMapper = sqlSession.getMapper(TeamMapper.class);
      teamMapper.updateByPrimaryKeySelective(team);

      TeamMemberMapper teamMemberMapper = sqlSession.getMapper(TeamMemberMapper.class);
      Map<String, Object> where = new HashMap<>();
      where.put("teamId", team.getId());

      // Take two lists of difference
      List<TeamMember> oldTeamMembers = teamMemberMapper.selectAll(where);
      List<String> oldMembers = new ArrayList<>();
      for (TeamMember oldTeamMember : oldTeamMembers) {
        oldMembers.add(oldTeamMember.getMember());
      }

      List<TeamMember> newTeamMembers = team.getCollaborators();
      List<String> newMembers = new ArrayList<>();
      for (TeamMember newTeamMember : newTeamMembers) {
        newMembers.add(newTeamMember.getMember());
      }

      for (TeamMember oldTeamMember : oldTeamMembers) {
        if (!newMembers.contains(oldTeamMember.getMember())) {
          teamMemberMapper.deleteByPrimaryKey(oldTeamMember.getId());
        }
      }

      for (TeamMember newTeamMember : newTeamMembers) {
        if (!oldMembers.contains(newTeamMember.getMember())) {
          // TODO(zhulinhao)：teamId Send it by the front desk, here there is no assignment
          newTeamMember.setTeamId(team.getId());
          newTeamMember.setTeamName(team.getTeamName());
          teamMemberMapper.insert(newTeamMember);
        }
      }

      // Updates all team_name of corresponding members in the teamMember table
      TeamMember teamMember = new TeamMember();
      teamMember.setTeamName(team.getTeamName());
      teamMember.setTeamId(team.getId());
      teamMemberMapper.updateSelective(teamMember);

      sqlSession.commit();
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new Exception(e);
    }
    return true;
  }

  public boolean delete(String id) throws Exception {
    LOG.info("delete({})", id);
    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      TeamMapper teamMapper = sqlSession.getMapper(TeamMapper.class);
      teamMapper.deleteByPrimaryKey(id);

      TeamMemberMapper teamMemberMapper = sqlSession.getMapper(TeamMemberMapper.class);
      TeamMember teamMember = new TeamMember();
      teamMember.setTeamId(id);
      teamMemberMapper.deleteSelective(teamMember);
      sqlSession.commit();
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new Exception(e);
    }
    return true;
  }

}
