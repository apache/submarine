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
import org.apache.submarine.server.model.database.entities.ModelVersionTagEntity;
import org.apache.submarine.server.database.model.mappers.ModelVersionTagMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class ModelVersionTagService {

  private static final Logger LOG = LoggerFactory.getLogger(ModelVersionTagService.class);

  public void insert(ModelVersionTagEntity modelVersionTag)
          throws SubmarineRuntimeException {
    LOG.info("Model Version Tag insert name:" + modelVersionTag.getName() + ", tag:" +
            modelVersionTag.getTag());
    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      ModelVersionTagMapper mapper = sqlSession.getMapper(ModelVersionTagMapper.class);
      mapper.insert(modelVersionTag);
      sqlSession.commit();
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new SubmarineRuntimeException("Unable to insert model version tag entity to database");
    }
  }

  public void delete(ModelVersionTagEntity modelVersionTag)
          throws SubmarineRuntimeException {
    LOG.info("Model Version Tag delete name:" + modelVersionTag.getName() + ", tag:" +
            modelVersionTag.getTag());
    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      ModelVersionTagMapper mapper = sqlSession.getMapper(ModelVersionTagMapper.class);
      mapper.delete(modelVersionTag);
      sqlSession.commit();
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new SubmarineRuntimeException("Unable to delete model version tag entity to database");
    }
  }
}
