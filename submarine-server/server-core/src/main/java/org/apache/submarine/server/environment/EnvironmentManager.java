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

import org.apache.submarine.commons.utils.exception.SubmarineRuntimeException;
import org.apache.submarine.server.api.environment.Environment;
import org.apache.submarine.server.api.spec.EnvironmentSpec;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Environment Management
 */
public class EnvironmentManager {
  
  private static final Logger LOG =
      LoggerFactory.getLogger(EnvironmentManager.class);

  private static volatile EnvironmentManager manager;

  private final ConcurrentMap<String, Environment> cachedEnvironments =
      new ConcurrentHashMap<>();

  private final AtomicInteger environmentId = new AtomicInteger(0);

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
    Environment env = new Environment();
    env.setName(spec.getName());
    env.setEnvironmentId(environmentId.incrementAndGet());
    env.setEnvironmentSpec(spec);
    cachedEnvironments.putIfAbsent(spec.getName(), env);
    return env;
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
    isEnvironmentExists(name);
    checkSpec(spec);
    Environment env = cachedEnvironments.get(name);
    env.setEnvironmentSpec(spec);
    cachedEnvironments.putIfAbsent(name, env);
    return env;
  }
  
  /**
   * Delete environment
   * @param name Name of the environment
   * @return Environment environment
   * @throws SubmarineRuntimeException the service error
   */
  public Environment deleteEnvironment(String name)
      throws SubmarineRuntimeException {
    isEnvironmentExists(name);
    Environment environment = cachedEnvironments.remove(name);
    return environment;
  }
  
  /**
   * Get Environment
   * @param name Name of the environment
   * @return Environment environment
   * @throws SubmarineRuntimeException the service error
   */
  public Environment getEnvironment(String name)
      throws SubmarineRuntimeException {
    isEnvironmentExists(name);
    Environment environment = cachedEnvironments.get(name);
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

  private void isEnvironmentExists(String name)
      throws SubmarineRuntimeException {
    if (name == null || !cachedEnvironments.containsKey(name)) {
      throw new SubmarineRuntimeException(Status.NOT_FOUND.getStatusCode(),
          "Environment not found.");
    }
  }
}
