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

package org.apache.submarine.server.environment;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.atomic.AtomicInteger;

import javax.ws.rs.core.Response.Status;

import org.apache.hadoop.hive.metastore.api.MetaException;
import org.apache.ibatis.session.SqlSession;
import org.apache.submarine.commons.metastore.SubmarineMetaStore;
import org.apache.submarine.commons.utils.SubmarineConfiguration;
import org.apache.submarine.commons.utils.exception.SubmarineRuntimeException;
import org.apache.submarine.server.api.environment.Environment;
import org.apache.submarine.server.api.spec.EnvironmentSpec;
import org.apache.submarine.server.database.utils.MyBatisUtil;
import org.apache.submarine.server.environment.database.entity.EnvironmentEntity;
import org.apache.submarine.server.environment.database.mappers.EnvironmentMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

/**
 * Environment Management
 */
public class EnvironmentManager {
  
  private static final Logger LOG =
      LoggerFactory.getLogger(EnvironmentManager.class);

  private static volatile EnvironmentManager manager;
  
  /**
   * Get the singleton instance
   * @return object
   */
  public static EnvironmentManager getInstance() {
    if (manager == null) {
      synchronized (EnvironmentManager.class) {
        if (manager == null) {
          manager = new EnvironmentManager();
        }
      }
    }
    return manager;
  }

  private EnvironmentManager() {
    
  }

  /**
   * Create Environment
   * @param spec environment spec
   * @return Environment environment
   * @throws SubmarineRuntimeException the service error
   * @throws MetaException 
   */
  public Environment createEnvironment(EnvironmentSpec spec)
      throws SubmarineRuntimeException {
    checkSpec(spec);
    EnvironmentEntity entity = new EnvironmentEntity();
    entity.setEnvironmentName(spec.getName());
    entity.setEnvironmentSpec(
        new GsonBuilder().disableHtmlEscaping().create().toJson(spec));
    
    LOG.info("add({})", entity.toString());
    try (SqlSession sqlSession = MyBatisUtil.getMetastoreSqlSession()) {
      EnvironmentMapper environmentMapper =
          sqlSession.getMapper(EnvironmentMapper.class);
      int environmentId = environmentMapper.insert(entity);
      sqlSession.commit();

      Environment env = new Environment();
      env.setName(spec.getName());
      env.setEnvironmentId(environmentId);
      env.setEnvironmentSpec(spec);
      return env;
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new SubmarineRuntimeException(Status.BAD_REQUEST.getStatusCode(),
          "Unable to process the environment spec.");
    }
  }
  
  /**
   * Update environment
   * @param name Name of the environment
   * @param spec environment spec
   * @return Environment environment
   * @throws SubmarineRuntimeException the service error
   */
  public Environment updateEnvironment(String name, EnvironmentSpec spec)
      throws SubmarineRuntimeException {
    Environment env = getEnvironmentDetails(name);
    if (env == null) {
      throw new SubmarineRuntimeException(Status.NOT_FOUND.getStatusCode(),
          "Environment not found.");
    }
    checkSpec(spec);
    
    EnvironmentEntity entity = new EnvironmentEntity();
    entity.setEnvironmentName(spec.getName());
    entity.setEnvironmentSpec(
        new GsonBuilder().disableHtmlEscaping().create().toJson(spec));
    try (SqlSession sqlSession = MyBatisUtil.getMetastoreSqlSession()) {
      EnvironmentMapper environmentMapper =
          sqlSession.getMapper(EnvironmentMapper.class);
      int environmentId = environmentMapper.update(entity);
      sqlSession.commit();

      Environment updatedEnvironment = new Environment();
      updatedEnvironment.setName(spec.getName());
      updatedEnvironment.setEnvironmentId(environmentId);
      updatedEnvironment.setEnvironmentSpec(spec);
      return updatedEnvironment;
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new SubmarineRuntimeException(Status.BAD_REQUEST.getStatusCode(),
          "Unable to process the environment spec.");
    }
  }
  
  /**
   * Delete environment
   * @param name Name of the environment
   * @return Environment environment
   * @throws SubmarineRuntimeException the service error
   */
  public Environment deleteEnvironment(String name)
      throws SubmarineRuntimeException {
    Environment env = getEnvironmentDetails(name);
    if (env == null) {
      throw new SubmarineRuntimeException(Status.NOT_FOUND.getStatusCode(),
          "Environment not found.");
    }
    
    try (SqlSession sqlSession = MyBatisUtil.getMetastoreSqlSession()) {
      EnvironmentMapper environmentMapper =
          sqlSession.getMapper(EnvironmentMapper.class);
      environmentMapper.delete(name);
      sqlSession.commit();
      return env;
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new SubmarineRuntimeException(Status.BAD_REQUEST.getStatusCode(),
          "Unable to delete the environment.");
    }
  }
  
  /**
   * Get Environment
   * @param name Name of the environment
   * @return Environment environment
   * @throws SubmarineRuntimeException the service error
   */
  public Environment getEnvironment(String name)
      throws SubmarineRuntimeException {
    Environment environment = getEnvironmentDetails(name);
    if (environment == null) {
      throw new SubmarineRuntimeException(Status.NOT_FOUND.getStatusCode(),
          "Environment not found.");
    }
    return environment;
  }

  /**
   * List environments
   * @param status environment status, if null will return all status
   * @return environment list
   * @throws SubmarineRuntimeException the service error
   */
  public List<Environment> listEnvironments(String status)
      throws SubmarineRuntimeException {
    List<Environment> environmentList = new ArrayList<>();
    return environmentList;
  }

  private void checkSpec(EnvironmentSpec spec)
      throws SubmarineRuntimeException {
    if (spec == null) {
      throw new SubmarineRuntimeException(Status.OK.getStatusCode(),
          "Invalid environment spec.");
    }
  }

  private Environment getEnvironmentDetails(String name)
      throws SubmarineRuntimeException {

    Environment env = null;
    try (SqlSession sqlSession = MyBatisUtil.getMetastoreSqlSession()) {
      EnvironmentMapper environmentMapper =
          sqlSession.getMapper(EnvironmentMapper.class);
      EnvironmentEntity environmentEntity = environmentMapper.select(name);

      if (environmentEntity != null) {
        env = new Environment();
        env.setName(environmentEntity.getEnvironmentName());
        env.setEnvironmentSpec(new Gson().fromJson(
            environmentEntity.getEnvironmentSpec(), EnvironmentSpec.class));
      }
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new SubmarineRuntimeException(Status.BAD_REQUEST.getStatusCode(),
          "Unable to get the environment details.");
    }
    return env;
  }
}
