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

package org.apache.submarine.server.security.common;

import org.apache.commons.lang3.StringUtils;
import org.apache.submarine.commons.utils.SubmarineConfiguration;

public class CommonConfig {

  public static final String LOGOUT_ENDPOINT = "/auth/logout";

  public static final String SUBMARINE_AUTH_MAX_AGE_ENV = "SUBMARINE_AUTH_MAX_AGE";
  public static final String SUBMARINE_AUTH_MAX_AGE_PROP = "submarine.auth.maxAge";

  public static final String AUTH_HEADER = "Authorization";

  public static final String AGENT_HEADER = "User-Agent";
  public static final String PYTHON_USER_AGENT;

  public static final int MAX_AGE;

  static {
    SubmarineConfiguration conf = SubmarineConfiguration.getInstance();
    MAX_AGE = conf.getInt(SUBMARINE_AUTH_MAX_AGE_ENV, SUBMARINE_AUTH_MAX_AGE_PROP,
            60 * 60 * 24);

    String version = System.getenv("SUBMARINE_VERSION");
    if (StringUtils.isBlank(version)) version = "0.7.0-SNAPSHOT";
    PYTHON_USER_AGENT = String.format("OpenAPI-Generator/%s/python", version);
  }

}
