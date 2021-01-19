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

package org.apache.submarine.server.experiment.database;

import org.apache.ibatis.session.SqlSession;
import org.apache.submarine.commons.utils.exception.SubmarineRuntimeException;
import org.apache.submarine.server.database.utils.MyBatisUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

public class ExperimentService {

  private static final Logger LOG = LoggerFactory.getLogger(ExperimentService.class);

  public List<ExperimentEntity> selectAll() throws SubmarineRuntimeException {
    LOG.info("Experiment selectAll");
    List<ExperimentEntity> entities;
    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      ExperimentMapper mapper = sqlSession.getMapper(ExperimentMapper.class);
      entities = mapper.selectAll();
      sqlSession.commit();
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new SubmarineRuntimeException("Unable to get experiment entities from database");
    }
    return entities;
  }

  public ExperimentEntity select(String id) throws SubmarineRuntimeException {
    LOG.info("Experiment select");
    ExperimentEntity entity;
    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      ExperimentMapper mapper = sqlSession.getMapper(ExperimentMapper.class);
      entity = mapper.select(id);
      sqlSession.commit();
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new SubmarineRuntimeException("Unable to get experiment entity from database");
    }
    return entity;
  }

  public boolean insert(ExperimentEntity experiment) throws SubmarineRuntimeException {
    LOG.info("Experiment insert");
    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      ExperimentMapper mapper = sqlSession.getMapper(ExperimentMapper.class);
      mapper.insert(experiment);
      sqlSession.commit();
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new SubmarineRuntimeException("Unable to insert experiment entity to database");
    }
    return true;
  }

  public boolean update(ExperimentEntity experiment) throws SubmarineRuntimeException {
    LOG.info("Experiment update");
    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      ExperimentMapper mapper = sqlSession.getMapper(ExperimentMapper.class);
      mapper.update(experiment);
      sqlSession.commit();
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new SubmarineRuntimeException("Unable to update experiment entity in database");
    }
    return true;
  }

  public boolean delete(String id) throws SubmarineRuntimeException {
    LOG.info("Experiment delete");
    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      ExperimentMapper mapper = sqlSession.getMapper(ExperimentMapper.class);
      mapper.delete(id);
      sqlSession.commit();
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new SubmarineRuntimeException("Unable to delete experiment entity from database");
    }
    return true;
  }
}
