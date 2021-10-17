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

package org.apache.submarine.server.model.database.service;

import org.apache.ibatis.session.SqlSession;
import org.apache.submarine.commons.utils.exception.SubmarineRuntimeException;
import org.apache.submarine.server.database.utils.MyBatisUtil;
import org.apache.submarine.server.model.database.entities.ModelVersionEntity;
import org.apache.submarine.server.model.database.mappers.ModelVersionMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

public class ModelVersionService {

  private static final Logger LOG = LoggerFactory.getLogger(ModelVersionService.class);

  public List<ModelVersionEntity> selectAllVersions(String name) throws SubmarineRuntimeException {
    LOG.info("Model Version select all versions:" + name);
    List<ModelVersionEntity> modelVersionEntities;
    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      ModelVersionMapper mapper = sqlSession.getMapper(ModelVersionMapper.class);
      modelVersionEntities = mapper.selectAllVersions(name);
      sqlSession.commit();
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new SubmarineRuntimeException("Unable to get model version from database");
    }
    return modelVersionEntities;
  }

  public ModelVersionEntity select(String name, Integer version) throws SubmarineRuntimeException {
    LOG.info("Model Version select:" + name + " " + version.toString());
    ModelVersionEntity modelVersionEntity;
    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      ModelVersionMapper mapper = sqlSession.getMapper(ModelVersionMapper.class);
      modelVersionEntity = mapper.select(name, version);
      sqlSession.commit();
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new SubmarineRuntimeException("Unable to get model version from database");
    }
    return modelVersionEntity;
  }
  public ModelVersionEntity selectWithTag(String name, Integer version) throws SubmarineRuntimeException {
    LOG.info("Model Version select with tag:" + name + " " + version.toString());
    ModelVersionEntity modelVersionEntity;
    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      ModelVersionMapper mapper = sqlSession.getMapper(ModelVersionMapper.class);
      modelVersionEntity = mapper.selectWithTag(name, version);
      sqlSession.commit();
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new SubmarineRuntimeException("Unable to get model version from database");
    }
    return modelVersionEntity;
  }

  public void insert(ModelVersionEntity modelVersion) throws SubmarineRuntimeException {
    LOG.info("Model Version insert " + modelVersion.getName());
    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      ModelVersionMapper mapper = sqlSession.getMapper(ModelVersionMapper.class);
      mapper.insert(modelVersion);
      sqlSession.commit();
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new SubmarineRuntimeException("Unable to insert model version from database");
    }
  }

  public void update(ModelVersionEntity modelVersion) throws SubmarineRuntimeException {
    LOG.info("Model Version update " + modelVersion.getName());
    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      ModelVersionMapper mapper = sqlSession.getMapper(ModelVersionMapper.class);
      mapper.update(modelVersion);
      sqlSession.commit();
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new SubmarineRuntimeException("Unable to update model version from database");
    }
  }

  public void delete(String name, Integer version) throws SubmarineRuntimeException {
    LOG.info("Model Version delete name:" + name + ", version:" + version.toString());
    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      ModelVersionMapper mapper = sqlSession.getMapper(ModelVersionMapper.class);
      mapper.delete(name, version);
      sqlSession.commit();
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new SubmarineRuntimeException("Unable to delete model version from database");
    }
  }
}
