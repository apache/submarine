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
    LOG.info("queryDictList owner:{}, column:{}, order:{}, pageNo:{}, pageSize:{}",
        owner, column, order, pageNo, pageSize);

    List<Team> list = null;
    SqlSession sqlSession = null;
    try {
      sqlSession = MyBatisUtil.getSqlSession();
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
    } finally {
      sqlSession.close();
    }
    return list;
  }

  public boolean add(Team team) throws Exception {
    LOG.info("add({})", team.toString());

    SqlSession sqlSession = null;
    try {
      sqlSession = MyBatisUtil.getSqlSession();
      TeamMapper teamMapper = sqlSession.getMapper(TeamMapper.class);
      teamMapper.insert(team);

      TeamMemberMapper teamMemberMapper = sqlSession.getMapper(TeamMemberMapper.class);
      // add teamMember, when add team, should insert 'Collaborators' to team_member
      List<TeamMember> list = team.getCollaborators();
      for (TeamMember teamMember : list) {
        // todo: teamMember's member is sys_user's id now.
        teamMember.setTeamId(team.getId());
        teamMemberMapper.insert(teamMember);
      }

      sqlSession.commit();
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new Exception(e);
    } finally {
      sqlSession.close();
    }
    return true;
  }

  public boolean updateByPrimaryKeySelective(Team team) throws Exception {
    LOG.info("updateByPrimaryKeySelective({})", team.toString());

    SqlSession sqlSession = null;
    try {
      sqlSession = MyBatisUtil.getSqlSession();
      TeamMapper teamMapper = sqlSession.getMapper(TeamMapper.class);
      teamMapper.updateByPrimaryKeySelective(team);

      TeamMemberMapper teamMemberMapper = sqlSession.getMapper(TeamMemberMapper.class);
      Map<String, Object> where = new HashMap<>();
      where.put("teamId", team.getId());

      // Take two lists of difference
      List<TeamMember> old_teamMembers = teamMemberMapper.selectAll(where);
      List<TeamMember> curr_teamMembers = team.getCollaborators();

      for (TeamMember old : old_teamMembers) {
        if (!curr_teamMembers.contains(old)) {
          teamMemberMapper.deleteByPrimaryKey(old.getId());
        }
      }

      for (TeamMember curr : curr_teamMembers) {
        if (!old_teamMembers.contains(curr)) {
          // todo：teamId Send it by the front desk, here there is no assignment
          curr.setTeamId(team.getId());
          curr.setTeamName(team.getTeamName());
          teamMemberMapper.insert(curr);
        }
      }

      sqlSession.commit();
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new Exception(e);
    } finally {
      sqlSession.close();
    }
    return true;
  }

  public boolean delete(String id) throws Exception {
    LOG.info("delete({})", id);
    SqlSession sqlSession = null;
    try {
      sqlSession = MyBatisUtil.getSqlSession();
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
    } finally {
      sqlSession.close();
    }
    return true;
  }

}
