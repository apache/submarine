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

import org.apache.ibatis.session.RowBounds;
import org.apache.ibatis.session.SqlSession;
import org.apache.submarine.server.database.utils.MyBatisUtil;
import org.apache.submarine.server.database.workbench.entity.TeamEntity;
import org.apache.submarine.server.database.workbench.entity.TeamMemberEntity;
import org.apache.submarine.server.database.workbench.mappers.TeamMapper;
import org.apache.submarine.server.database.workbench.mappers.TeamMemberMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class TeamService {
  private static final Logger LOG = LoggerFactory.getLogger(TeamService.class);

  public List<TeamEntity> queryPageList(String owner,
                                        String column,
                                        String order,
                                        int pageNo,
                                        int pageSize) throws Exception {
    LOG.info("queryPageList owner:{}, column:{}, order:{}, pageNo:{}, pageSize:{}",
        owner, column, order, pageNo, pageSize);

    List<TeamEntity> list = null;
    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      TeamMapper teamMapper = sqlSession.getMapper(TeamMapper.class);
      Map<String, Object> where = new HashMap<>();
      where.put("owner", owner);
      where.put("column", column);
      where.put("order", order);
      list = teamMapper.selectAll(where, new RowBounds(pageNo, pageSize));

      TeamMemberMapper teamMemberMapper = sqlSession.getMapper(TeamMemberMapper.class);
      // query from team_member table, and set to team
      for (TeamEntity team : list) {
        Map<String, Object> whereMember = new HashMap<>();
        whereMember.put("teamId", team.getId());
        List<TeamMemberEntity> teamMembers = teamMemberMapper.selectAll(whereMember);
        for (TeamMemberEntity teamMember : teamMembers) {
          team.addCollaborator(teamMember);
        }
      }
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new Exception(e);
    }
    return list;
  }

  public boolean add(TeamEntity team) throws Exception {
    LOG.info("add({})", team.toString());

    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      TeamMapper teamMapper = sqlSession.getMapper(TeamMapper.class);
      teamMapper.insert(team);

      TeamMemberMapper teamMemberMapper = sqlSession.getMapper(TeamMemberMapper.class);
      // add teamMember, when add team, should insert 'Collaborators' to team_member
      List<TeamMemberEntity> list = team.getCollaborators();
      for (TeamMemberEntity teamMember : list) {
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

  public boolean updateByPrimaryKeySelective(TeamEntity team) throws Exception {
    LOG.info("updateByPrimaryKeySelective({})", team.toString());

    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      TeamMapper teamMapper = sqlSession.getMapper(TeamMapper.class);
      teamMapper.updateByPrimaryKeySelective(team);

      TeamMemberMapper teamMemberMapper = sqlSession.getMapper(TeamMemberMapper.class);
      Map<String, Object> where = new HashMap<>();
      where.put("teamId", team.getId());

      // Take two lists of difference
      List<TeamMemberEntity> oldTeamMembers = teamMemberMapper.selectAll(where);
      List<String> oldMembers = new ArrayList<>();
      for (TeamMemberEntity oldTeamMember : oldTeamMembers) {
        oldMembers.add(oldTeamMember.getMember());
      }

      List<TeamMemberEntity> newTeamMembers = team.getCollaborators();
      List<String> newMembers = new ArrayList<>();
      for (TeamMemberEntity newTeamMember : newTeamMembers) {
        newMembers.add(newTeamMember.getMember());
      }

      for (TeamMemberEntity oldTeamMember : oldTeamMembers) {
        if (!newMembers.contains(oldTeamMember.getMember())) {
          teamMemberMapper.deleteByPrimaryKey(oldTeamMember.getId());
        }
      }

      for (TeamMemberEntity newTeamMember : newTeamMembers) {
        if (!oldMembers.contains(newTeamMember.getMember())) {
          // TODO(zhulinhao)：teamId Send it by the front desk, here there is no assignment
          newTeamMember.setTeamId(team.getId());
          newTeamMember.setTeamName(team.getTeamName());
          teamMemberMapper.insert(newTeamMember);
        }
      }

      // Updates all team_name of corresponding members in the teamMember table
      TeamMemberEntity teamMember = new TeamMemberEntity();
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
      TeamMemberEntity teamMember = new TeamMemberEntity();
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
