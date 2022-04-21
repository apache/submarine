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

import org.apache.ibatis.session.SqlSession;
import org.apache.submarine.server.database.utils.MyBatisUtil;
import org.apache.submarine.server.database.workbench.entity.ProjectFilesEntity;
import org.apache.submarine.server.database.workbench.mappers.ProjectFilesMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class ProjectFilesService {
  private static final Logger LOG = LoggerFactory.getLogger(ProjectFilesService.class);

  public List<ProjectFilesEntity> queryList(String projectId) throws Exception {
    LOG.info("queryPageList projectId:{}", projectId);

    List<ProjectFilesEntity> list = null;
    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      ProjectFilesMapper projectFilesMapper = sqlSession.getMapper(ProjectFilesMapper.class);
      Map<String, Object> where = new HashMap<>();
      where.put("projectId", projectId);
      list = projectFilesMapper.selectAll(where);
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new Exception(e);
    }
    return list;
  }

}
