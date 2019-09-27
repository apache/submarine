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

import org.apache.ibatis.session.SqlSession;
import org.apache.submarine.database.MyBatisUtil;
import org.apache.submarine.database.entity.ProjectFiles;
import org.apache.submarine.database.mappers.ProjectFilesMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class ProjectFilesService {
  private static final Logger LOG = LoggerFactory.getLogger(ProjectFilesService.class);

  public List<ProjectFiles> queryList(String projectId) throws Exception {
    LOG.info("queryPageList projectId:{}", projectId);

    List<ProjectFiles> list = null;
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
