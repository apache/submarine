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
import org.apache.submarine.server.database.utils.MyBatisUtil;
import org.apache.submarine.server.database.model.entities.RegisteredModelEntity;
import org.apache.submarine.server.database.model.mappers.RegisteredModelMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

public class RegisteredModelService {

  private static class RegisteredModelServiceHolder {
    private static RegisteredModelService service = new RegisteredModelService();
  }

  public static RegisteredModelService getInstance() {
    return RegisteredModelService.RegisteredModelServiceHolder.service;
  }

  private static final Logger LOG = LoggerFactory.getLogger(RegisteredModelService.class);

  public List<RegisteredModelEntity> selectAll() throws SubmarineRuntimeException {
    LOG.info("Registered model selectAll");
    List<RegisteredModelEntity> registeredModels;
    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      RegisteredModelMapper mapper = sqlSession.getMapper(RegisteredModelMapper.class);
      registeredModels = mapper.selectAll();
      sqlSession.commit();
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new SubmarineRuntimeException("Unable to get registered models from database");
    }
    return registeredModels;
  }

  public RegisteredModelEntity select(String name) throws SubmarineRuntimeException {
    LOG.info("Registered Model select:" + name);
    RegisteredModelEntity registeredModel;
    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      RegisteredModelMapper mapper = sqlSession.getMapper(RegisteredModelMapper.class);
      registeredModel = mapper.select(name);
      sqlSession.commit();
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new SubmarineRuntimeException("Unable to get registered model name entity from database");
    }
    return registeredModel;
  }

  public RegisteredModelEntity selectWithTag(String name) throws SubmarineRuntimeException {
    LOG.info("Registered Model select with tag:" + name);
    RegisteredModelEntity registeredModel;
    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      RegisteredModelMapper mapper = sqlSession.getMapper(RegisteredModelMapper.class);
      registeredModel = mapper.selectWithTag(name);
      sqlSession.commit();
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new SubmarineRuntimeException("Unable to get registered model name entity from database");
    }
    return registeredModel;
  }

  public void insert(RegisteredModelEntity registeredModel) throws SubmarineRuntimeException {
    LOG.info("Registered Model insert " + registeredModel.getName());
    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      RegisteredModelMapper mapper = sqlSession.getMapper(RegisteredModelMapper.class);
      mapper.insert(registeredModel);
      sqlSession.commit();
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new SubmarineRuntimeException("Unable to insert registered model name entity to database");
    }
  }

  public void update(RegisteredModelEntity registeredModel) throws SubmarineRuntimeException {
    LOG.info("Registered Model update " + registeredModel.getName());
    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      RegisteredModelMapper mapper = sqlSession.getMapper(RegisteredModelMapper.class);
      mapper.update(registeredModel);
      sqlSession.commit();
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new SubmarineRuntimeException("Unable to update registered model name entity from database");
    }
  }

  public void rename(String name, String newName) throws SubmarineRuntimeException {
    LOG.info("Registered Model rename");
    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      RegisteredModelMapper mapper = sqlSession.getMapper(RegisteredModelMapper.class);
      mapper.rename(name, newName);
      sqlSession.commit();
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new SubmarineRuntimeException("Unable to rename registered model name from database");
    }
  }

  public void delete(String name) throws SubmarineRuntimeException {
    LOG.info("Registered Model delete " + name);
    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      RegisteredModelMapper mapper = sqlSession.getMapper(RegisteredModelMapper.class);
      mapper.delete(name);
      sqlSession.commit();
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new SubmarineRuntimeException("Unable to delete registered model entity from database");
    }
  }

  public void deleteAll() throws SubmarineRuntimeException {
    LOG.info("Registered Model delete all");
    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      RegisteredModelMapper mapper = sqlSession.getMapper(RegisteredModelMapper.class);
      mapper.deleteAll();
      sqlSession.commit();
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new SubmarineRuntimeException("Unable to delete all registered model entities from database");
    }
  }
}
