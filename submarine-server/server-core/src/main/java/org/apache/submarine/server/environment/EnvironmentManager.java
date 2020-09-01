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

import org.apache.ibatis.session.SqlSession;
import org.apache.submarine.commons.utils.exception.SubmarineRuntimeException;
import org.apache.submarine.server.SubmarineServer;
import org.apache.submarine.server.api.environment.Environment;
import org.apache.submarine.server.api.environment.EnvironmentId;
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

  private final AtomicInteger environmentIdCounter = new AtomicInteger(0);

  private static Boolean readedDB = true;

  /**
   * Environment Cache
   */
  private final ConcurrentMap<String, Environment> cachedEnvironments =
      new ConcurrentHashMap<>();

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
   */
  public Environment createEnvironment(EnvironmentSpec spec)
      throws SubmarineRuntimeException {
    checkSpec(spec);
    LOG.info("Create Environment using spec: " + spec.toString());
    return createOrUpdateEnvironment(spec, "c");
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
    LOG.info("Update Environment using spec: " + spec.toString());
    return createOrUpdateEnvironment(spec, "u");
  }

  private Environment createOrUpdateEnvironment(EnvironmentSpec spec,
      String operation) {
    EnvironmentEntity entity = new EnvironmentEntity();
    String environmentId = generateEnvironmentId().toString();
    entity.setId(environmentId);
    entity.setEnvironmentName(spec.getName());
    entity.setEnvironmentSpec(
        new GsonBuilder().disableHtmlEscaping().create().toJson(spec));
    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      EnvironmentMapper environmentMapper =
          sqlSession.getMapper(EnvironmentMapper.class);
      if (operation.equals("c")) {
        environmentMapper.insert(entity);
      } else {
        environmentMapper.update(entity);
      }
      sqlSession.commit();

      Environment environment = new Environment();
      environment.setEnvironmentId(EnvironmentId.fromString(environmentId));
      environment.setEnvironmentSpec(spec);

      // Update cache
      cachedEnvironments.putIfAbsent(spec.getName(), environment);

      return environment;
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new SubmarineRuntimeException(Status.BAD_REQUEST.getStatusCode(),
          "Unable to process the environment spec.");
    }
  }

  private EnvironmentId generateEnvironmentId() {
    return EnvironmentId.newInstance(SubmarineServer.getServerTimeStamp(),
        environmentIdCounter.incrementAndGet());
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

    LOG.info("Delete Environment for " + name);
    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      EnvironmentMapper environmentMapper =
          sqlSession.getMapper(EnvironmentMapper.class);
      environmentMapper.delete(name);
      sqlSession.commit();

      // Invalidate cache
      cachedEnvironments.remove(name);
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
    List<Environment> environmentList =
        new ArrayList<Environment>(cachedEnvironments.values());

    // Is it available in cache?
    if (this.readedDB == true) {
      try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
        EnvironmentMapper environmentMapper =
              sqlSession.getMapper(EnvironmentMapper.class);
        List<EnvironmentEntity> environmentEntitys = environmentMapper.selectAll();
        for (EnvironmentEntity environmentEntity : environmentEntitys) {
          if (environmentEntity != null) {
            Environment env = new Environment();
            env.setEnvironmentSpec(new Gson().fromJson(
                  environmentEntity.getEnvironmentSpec(), EnvironmentSpec.class));
            env.setEnvironmentId(
                  EnvironmentId.fromString(environmentEntity.getId()));
            environmentList.add(env);
            cachedEnvironments.put(env.getEnvironmentSpec().getName(), env);
          }
        }
      } catch (Exception e) {
        LOG.error(e.getMessage(), e);
        throw new SubmarineRuntimeException(Status.BAD_REQUEST.getStatusCode(),
              "Unable to get the environment list.");
      }
    }
    this.readedDB = false;
    return environmentList;
  }

  private void checkSpec(EnvironmentSpec spec)
      throws SubmarineRuntimeException {
    if (spec == null) {
      throw new SubmarineRuntimeException(Status.BAD_REQUEST.getStatusCode(),
          "Invalid environment spec.");
    }
  }

  private Environment getEnvironmentDetails(String name)
      throws SubmarineRuntimeException {

    // Is it available in cache?
    Environment env = cachedEnvironments.get(name);
    if (env != null) {
      return env;
    }

    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      EnvironmentMapper environmentMapper = sqlSession.getMapper(EnvironmentMapper.class);
      EnvironmentEntity environmentEntity = environmentMapper.select(name);

      if (environmentEntity != null) {
        env = new Environment();
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
