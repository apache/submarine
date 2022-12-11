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

import org.apache.submarine.commons.utils.SubmarineConfiguration;

import static org.apache.submarine.commons.utils.SubmarineConfVars.ConfVars;

public class CommonConfig {

  public static final String LOGOUT_ENDPOINT = "/auth/logout";
  public static final String AUTH_HEADER = "Authorization";
  public static final String BEARER_HEADER_PREFIX = "Bearer ";

  public static final int MAX_AGE;

  public static final String AGENT_HEADER = "User-Agent";
  // python sdk agent header (submarine-sdk/pysubmarine/submarine/client/api_client.py#93)
  // We only deal with front and server, py-sdk is not dealt with now
  public static final String PYTHON_USER_AGENT_REGREX = "^OpenAPI-Generator/[\\w\\-\\.]+/python$";

  static {
    SubmarineConfiguration conf = SubmarineConfiguration.getInstance();
    MAX_AGE = conf.getInt(ConfVars.SUBMARINE_AUTH_MAX_AGE_ENV);
  }

}
