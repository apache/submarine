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
package org.apache.submarine.server.workbench.database.service;

import org.apache.ibatis.session.SqlSession;
import org.apache.submarine.server.database.utils.MyBatisUtil;
import org.apache.submarine.server.workbench.database.entity.TeamMember;
import org.apache.submarine.server.workbench.database.mappers.TeamMemberMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class TeamMemberService {
  private static final Logger LOG = LoggerFactory.getLogger(TeamMemberService.class);

  public List<TeamMember> queryList(String teamId) throws Exception {
    LOG.info("queryList teamId:{}", teamId);

    List<TeamMember> list = null;
    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      TeamMemberMapper teamMemberMapper = sqlSession.getMapper(TeamMemberMapper.class);
      Map<String, Object> where = new HashMap<>();
      where.put("teamId", teamId);
      list = teamMemberMapper.selectAll(where);
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new Exception(e);
    }
    return list;
  }

  public boolean insertSelective(TeamMember teamMember) throws Exception {
    LOG.info("insertSelective({})", teamMember.toString());

    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      TeamMemberMapper teamMemberMapper = sqlSession.getMapper(TeamMemberMapper.class);
      teamMemberMapper.insertSelective(teamMember);

      sqlSession.commit();
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new Exception(e);
    }
    return true;
  }

  public void deleteByPrimaryKey(String id) throws Exception {
    LOG.info("deleteByPrimaryKey({})", id);

    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      TeamMemberMapper teamMemberMapper = sqlSession.getMapper(TeamMemberMapper.class);
      teamMemberMapper.deleteByPrimaryKey(id);
      sqlSession.commit();
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new Exception(e);
    }
  }
}
