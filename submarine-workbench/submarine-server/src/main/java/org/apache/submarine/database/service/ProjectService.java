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

import org.apache.ibatis.session.RowBounds;
import org.apache.ibatis.session.SqlSession;
import org.apache.submarine.database.MyBatisUtil;
import org.apache.submarine.database.entity.Project;
import org.apache.submarine.database.entity.ProjectFiles;
import org.apache.submarine.database.mappers.ProjectFilesMapper;
import org.apache.submarine.database.mappers.ProjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

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
    SqlSession sqlSession = null;
    try {
      sqlSession = MyBatisUtil.getSqlSession();
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
    } finally {
      sqlSession.close();
    }
    return list;
  }

  public boolean delete(String id) throws Exception {
    LOG.info("delete({})", id);
    SqlSession sqlSession = null;
    try {
      sqlSession = MyBatisUtil.getSqlSession();
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
    } finally {
      sqlSession.close();
    }
    return true;
  }

}
