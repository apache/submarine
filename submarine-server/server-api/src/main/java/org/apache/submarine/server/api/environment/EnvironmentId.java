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

package org.apache.submarine.server.api.environment;

import org.apache.submarine.commons.utils.AbstractUniqueIdGenerator;

/**
 * The unique id for environment. Formatter:
 * environment_${server_timestamp}_${counter} Such as:
 * environment__1577627710_0001
 */
public class EnvironmentId extends AbstractUniqueIdGenerator<EnvironmentId> {
  
  private static final String ENVIRONMENT_ID_PREFIX = "environment_";

  /**
   * Get the object of EnvironmentId.
   * 
   * @param environmentId Id of the environment string
   * @return object
   */
  public static EnvironmentId fromString(String environmentId) {
    if (environmentId == null) {
      return null;
    }
    String[] components = environmentId.split("\\_");
    if (components.length != 3) {
      return null;
    }
    return EnvironmentId.newInstance(Long.parseLong(components[1]),
        Integer.parseInt(components[2]));
  }

  /**
   * Ge the object of EnvironmentId.
   * 
   * @param serverTimestamp the timestamp when the server start
   * @param id count
   * @return object
   */
  public static EnvironmentId newInstance(long serverTimestamp, int id) {
    EnvironmentId experimentId = new EnvironmentId();
    experimentId.setServerTimestamp(serverTimestamp);
    experimentId.setId(id);
    return experimentId;
  }

  @Override
  public String toString() {
    StringBuilder sb = new StringBuilder(64);
    sb.append(ENVIRONMENT_ID_PREFIX).append(getServerTimestamp()).append("_");
    format(sb, getId());
    return sb.toString();
  }
}
