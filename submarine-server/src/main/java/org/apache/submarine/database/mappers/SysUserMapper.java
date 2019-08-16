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
package org.apache.submarine.database.mappers;

import org.apache.ibatis.session.SqlSession;
import org.apache.submarine.database.MyBatisUtil;
import org.apache.submarine.database.entity.SysUser;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashMap;
import java.util.Map;

public class SysUserMapper {
  private static final Logger LOG = LoggerFactory.getLogger(SysUserMapper.class);

  private static String GET_USER_BY_NAME_STATEMENT
      = "org.apache.submarine.database.mappers.SysUserMapper.getUserByName";

  public SysUser getUserByName(String name, String password) {
    SysUser sysUser = null;
    SqlSession sqlSession = MyBatisUtil.getSqlSessionFactory().openSession();
    try {
      HashMap<String, Object> mapParams = new HashMap<>();
      mapParams.put("name", name);
      mapParams.put("password", password);

      Map<String, Object> params = new HashMap<>();
      params.put("mapParams", mapParams);

      sysUser = sqlSession.selectOne(GET_USER_BY_NAME_STATEMENT, params);
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
    } finally {
      sqlSession.close();
    }

    return sysUser;
  }
}
