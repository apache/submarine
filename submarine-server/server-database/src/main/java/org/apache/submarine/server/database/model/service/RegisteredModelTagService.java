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

package org.apache.submarine.server.database.model.service;

import org.apache.ibatis.session.SqlSession;
import org.apache.submarine.commons.utils.exception.SubmarineRuntimeException;
import org.apache.submarine.server.database.database.utils.MyBatisUtil;
import org.apache.submarine.server.database.model.entities.RegisteredModelTagEntity;
import org.apache.submarine.server.database.model.mappers.RegisteredModelTagMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class RegisteredModelTagService {

  private static final Logger LOG = LoggerFactory.getLogger(RegisteredModelTagService.class);

  public void insert(RegisteredModelTagEntity registeredModelTag)
          throws SubmarineRuntimeException {
    LOG.info("Registered Model Tag insert name:" + registeredModelTag.getName() + ", tag:" +
            registeredModelTag.getTag());
    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      RegisteredModelTagMapper mapper = sqlSession.getMapper(RegisteredModelTagMapper.class);
      mapper.insert(registeredModelTag);
      sqlSession.commit();
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new SubmarineRuntimeException("Unable to insert registered model tag entity to database");
    }
  }

  public void delete(RegisteredModelTagEntity registeredModelTag)
          throws SubmarineRuntimeException {
    LOG.info("Registered Model Tag delete name:" + registeredModelTag.getName() + ", tag:" +
            registeredModelTag.getTag());
    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      RegisteredModelTagMapper mapper = sqlSession.getMapper(RegisteredModelTagMapper.class);
      mapper.delete(registeredModelTag);
      sqlSession.commit();
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new SubmarineRuntimeException("Unable to delete registered model tag from database");
    }
  }
}
