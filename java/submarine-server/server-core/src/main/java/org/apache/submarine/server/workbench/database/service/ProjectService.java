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

import org.apache.ibatis.session.RowBounds;
import org.apache.ibatis.session.SqlSession;
import org.apache.submarine.server.workbench.database.MyBatisUtil;
import org.apache.submarine.server.workbench.database.entity.Project;
import org.apache.submarine.server.workbench.database.entity.ProjectFiles;
import org.apache.submarine.server.workbench.database.mappers.ProjectFilesMapper;
import org.apache.submarine.server.workbench.database.mappers.ProjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class ProjectService {
  private static final Logger LOG = LoggerFactory.getLogger(ProjectService.class);

  public List<Project> queryPageList(String userName,
                                     String column,
                                     String order,
                                     int pageNo,
                                     int pageSize) throws Exception {
    LOG.info("queryPageList owner:{}, column:{}, order:{}, pageNo:{}, pageSize:{}",
        userName, column, order, pageNo, pageSize);

    List<Project> list = null;
    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      ProjectMapper projectMapper = sqlSession.getMapper(ProjectMapper.class);
      Map<String, Object> where = new HashMap<>();
      where.put("userName", userName);
      where.put("column", column);
      where.put("order", order);
      list = projectMapper.selectAll(where, new RowBounds(pageNo, pageSize));

      ProjectFilesMapper projectFilesMapper = sqlSession.getMapper(ProjectFilesMapper.class);
      // Query from project_files table, And set to Project Object
      for (Project project : list) {
        Map<String, Object> whereMember = new HashMap<>();
        whereMember.put("projectId", project.getId());
        List<ProjectFiles> projectFilesList = projectFilesMapper.selectAll(whereMember);
        for (ProjectFiles projectFiles : projectFilesList) {
          project.addProjectFilesList(projectFiles);
        }
      }
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new Exception(e);
    }
    return list;
  }

  public boolean add(Project project) throws Exception {
    LOG.info("add({})", project.toString());

    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      ProjectMapper projectMapper = sqlSession.getMapper(ProjectMapper.class);
      projectMapper.insert(project);

      ProjectFilesMapper projectFilesMapper = sqlSession.getMapper(ProjectFilesMapper.class);
      // add ProjectFiles, when add project, should insert 'ProjectFilesList' to ProjectFiles
      List<ProjectFiles> list = project.getProjectFilesList();
      for (ProjectFiles projectFiles : list) {
        // ProjectId needs to be obtained after the Project is inserted into the database
        projectFiles.setProjectId(project.getId());
        projectFilesMapper.insert(projectFiles);
      }

      sqlSession.commit();
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new Exception(e);
    }
    return true;
  }

  public boolean updateByPrimaryKeySelective(Project project) throws Exception {
    LOG.info("updateByPrimaryKeySelective({})", project.toString());
    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      ProjectMapper projectMapper = sqlSession.getMapper(ProjectMapper.class);
      projectMapper.updateByPrimaryKeySelective(project);

      ProjectFilesMapper projectFilesMapper = sqlSession.getMapper(ProjectFilesMapper.class);
      Map<String, Object> where = new HashMap<>();
      where.put("projectId", project.getId());
      // Take two lists of difference
      List<ProjectFiles> oldProjectFiles = projectFilesMapper.selectAll(where);
      List<String> oldProjectFilesId = new ArrayList<>();
      for (ProjectFiles oldProjectFile : oldProjectFiles) {
        oldProjectFilesId.add(oldProjectFile.getId());
      }
      List<ProjectFiles> currProjectFiles = project.getProjectFilesList();
      List<String> currProjectFilesId = new ArrayList<>();
      for (ProjectFiles currProjectFile : currProjectFiles) {
        currProjectFilesId.add(currProjectFile.getId());
      }

      for (ProjectFiles old : oldProjectFiles) {
        if (!currProjectFilesId.contains(old.getId())) {
          projectFilesMapper.deleteByPrimaryKey(old.getId());
        } else {
          for (ProjectFiles currProjectFile : currProjectFiles) {
            if (currProjectFile.getId() != null && currProjectFile.getId().equals(old.getId())) {
              projectFilesMapper.updateByPrimaryKeySelective(currProjectFile);
            }
          }
        }
      }
      for (ProjectFiles curr : currProjectFiles) {
        if (curr.getId() == null) {
          // TODO(zhulinhao)：The front desk should pass the projectId
          curr.setProjectId(project.getId());
          projectFilesMapper.insert(curr);
        }
      }

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
      ProjectMapper projectMapper = sqlSession.getMapper(ProjectMapper.class);
      projectMapper.deleteByPrimaryKey(id);

      ProjectFilesMapper projectFilesMapper = sqlSession.getMapper(ProjectFilesMapper.class);
      ProjectFiles projectFiles = new ProjectFiles();
      projectFiles.setProjectId(id);
      projectFilesMapper.deleteSelective(projectFiles);
      sqlSession.commit();
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new Exception(e);
    }
    return true;
  }

}
