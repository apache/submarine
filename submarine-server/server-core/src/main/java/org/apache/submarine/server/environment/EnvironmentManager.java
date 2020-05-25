package org.apache.submarine.server.environment;

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
import java.util.ArrayList;
import java.util.List;

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
  
  private static final Logger LOG = LoggerFactory.getLogger(EnvironmentManager.class);

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
   * @return object
   * @throws SubmarineRuntimeException the service error
   */
  public Environment createEnvironment(EnvironmentSpec spec) throws SubmarineRuntimeException {
    checkSpec(spec);
    return null;
  }
  
  /**
   * Update environment
   * @param id environment id
   * @param spec environment spec
   * @return object
   * @throws SubmarineRuntimeException the service error
   */
  public Environment updateEnvironment(String id, EnvironmentSpec spec) throws SubmarineRuntimeException {
    return null;
  }
  
  /**
   * Delete environment
   * @param id environment id
   * @return object
   * @throws SubmarineRuntimeException the service error
   */
  public Environment deleteEnvironment(String id) throws SubmarineRuntimeException {
    return null;
  }
  
  /**
   * Get Environment
   * @param id environment id
   * @return object
   * @throws SubmarineRuntimeException the service error
   */
  public Environment getEnvironment(String id) throws SubmarineRuntimeException {
    return null;
  }

  /**
   * List environments
   * @param status environment status, if null will return all status
   * @return environment list
   * @throws SubmarineRuntimeException the service error
   */
  public List<Environment> listEnvironments(String status) throws SubmarineRuntimeException {
    List<Environment> environmentList = new ArrayList<>();
    return environmentList;
  }

  private void checkSpec(EnvironmentSpec spec) throws SubmarineRuntimeException {
    if (spec == null) {
      throw new SubmarineRuntimeException(Status.OK.getStatusCode(), "Invalid job spec.");
    }
  }
}
