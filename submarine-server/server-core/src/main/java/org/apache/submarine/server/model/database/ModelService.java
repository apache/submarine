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

package org.apache.submarine.server.model.database;

import org.apache.ibatis.session.SqlSession;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.util.List;
import org.apache.submarine.commons.utils.exception.SubmarineRuntimeException;
import org.apache.submarine.server.database.utils.ModelBatisUtil;
import org.apache.submarine.server.model.database.entities.RegisteredModelNameEntity;
import org.apache.submarine.server.model.database.entities.ModelVersionEntity;
import org.apache.submarine.server.model.database.mappers.RegisteredModelNameMapper;
import org.apache.submarine.server.model.database.mappers.ModelVersionMapper;

public class ModelService {

  private static final Logger
      LOG = LoggerFactory.getLogger(org.apache.submarine.server.model.database.ModelService.class);

  public List<RegisteredModelNameEntity> selectAllRegisteredModelName() throws SubmarineRuntimeException {
    LOG.info("Registered Model Name selectAll");
    List<RegisteredModelNameEntity> entities;
    try (SqlSession sqlSession = ModelBatisUtil.getSqlSession()) {
      RegisteredModelNameMapper mapper = sqlSession.getMapper(RegisteredModelNameMapper.class);
      entities = mapper.selectAll();
      sqlSession.commit();
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new SubmarineRuntimeException("Unable to get registered model name entities from database");
    }
    return entities;
  }

  public RegisteredModelNameEntity selectRegisteredModelName(String name) throws SubmarineRuntimeException {
    LOG.info("Registered Model Name select " + name);
    RegisteredModelNameEntity entity;
    try (SqlSession sqlSession = ModelBatisUtil.getSqlSession()) {
      RegisteredModelNameMapper mapper = sqlSession.getMapper(RegisteredModelNameMapper.class);
      entity = mapper.select(name);
      sqlSession.commit();
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new SubmarineRuntimeException("Unable to get registered model name entity from database");
    }
    return entity;
  }

  public boolean insertRegisteredModelName(RegisteredModelNameEntity entity)
      throws SubmarineRuntimeException {
    LOG.info("Registered Model Name insert " + entity.getName());
    try (SqlSession sqlSession = ModelBatisUtil.getSqlSession()) {
      RegisteredModelNameMapper mapper = sqlSession.getMapper(RegisteredModelNameMapper.class);
      mapper.insert(entity);
      sqlSession.commit();
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new SubmarineRuntimeException("Unable to insert registered model name entity from database");
    }
    return true;
  }

  public boolean updateRegisteredModelName(RegisteredModelNameEntity entity)
      throws SubmarineRuntimeException {
    LOG.info("Registered Model Name update " + entity.getName());
    try (SqlSession sqlSession = ModelBatisUtil.getSqlSession()) {
      RegisteredModelNameMapper mapper = sqlSession.getMapper(RegisteredModelNameMapper.class);
      mapper.update(entity);
      sqlSession.commit();
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new SubmarineRuntimeException("Unable to update registered model name entity from database");
    }
    return true;
  }

  public boolean deleteRegisteredModelName(String name) throws SubmarineRuntimeException {
    LOG.info("Registered Model Name delete " + name);
    try (SqlSession sqlSession = ModelBatisUtil.getSqlSession()) {
      RegisteredModelNameMapper mapper = sqlSession.getMapper(RegisteredModelNameMapper.class);
      mapper.delete(name);
      sqlSession.commit();
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new SubmarineRuntimeException("Unable to delete registered model name entity from database");
    }
    return true;
  }

  public List<ModelVersionEntity> listModelVersion(String name) throws SubmarineRuntimeException {
    LOG.info("Model Version list " + name);
    List<ModelVersionEntity> entities;
    try (SqlSession sqlSession = ModelBatisUtil.getSqlSession()) {
      ModelVersionMapper mapper = sqlSession.getMapper(ModelVersionMapper.class);
      entities = mapper.list(name);
      sqlSession.commit();
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new SubmarineRuntimeException("Unable to get model version entities from database");
    }
    return entities;
  }

  public ModelVersionEntity selectModelVersion(String name, Integer version)
      throws SubmarineRuntimeException {
    LOG.info("Model Version select " + name + " " + version.toString());
    ModelVersionEntity entity = new ModelVersionEntity();
    entity.setName(name);
    entity.setVersion(version);
    try (SqlSession sqlSession = ModelBatisUtil.getSqlSession()) {
      ModelVersionMapper mapper = sqlSession.getMapper(ModelVersionMapper.class);
      entity = mapper.select(entity);
      sqlSession.commit();
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new SubmarineRuntimeException("Unable to get model version entity from database");
    }
    return entity;
  }

  public boolean insertModelVersion(ModelVersionEntity entity) throws SubmarineRuntimeException {
    LOG.info("Model Version insert " + entity.getName());
    try (SqlSession sqlSession = ModelBatisUtil.getSqlSession()) {
      ModelVersionMapper mapper = sqlSession.getMapper(ModelVersionMapper.class);
      mapper.insert(entity);
      sqlSession.commit();
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new SubmarineRuntimeException("Unable to insert model version entity from database");
    }
    return true;
  }


  public boolean deleteModelVersion(String name, Integer version) throws SubmarineRuntimeException {
    LOG.info("Model Version delete " + name + " " + version.toString());
    ModelVersionEntity entity = new ModelVersionEntity();
    entity.setName(name);
    entity.setVersion(version);
    try (SqlSession sqlSession = ModelBatisUtil.getSqlSession()) {
      ModelVersionMapper mapper = sqlSession.getMapper(ModelVersionMapper.class);
      mapper.delete(entity);
      sqlSession.commit();
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new SubmarineRuntimeException("Unable to delete model version entity from database");
    }
    return true;
  }

  public boolean deleteAllModelVersion(String name) throws SubmarineRuntimeException {
    LOG.info("Model Version delete all " + name);
    try (SqlSession sqlSession = ModelBatisUtil.getSqlSession()) {
      ModelVersionMapper mapper = sqlSession.getMapper(ModelVersionMapper.class);
      mapper.deleteAll(name);
      sqlSession.commit();
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new SubmarineRuntimeException("Unable to delete model version entity from database");
    }
    return true;
  }

}
