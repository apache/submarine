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

package org.apache.submarine.server.security.oidc;

import org.apache.submarine.commons.utils.SubmarineConfiguration;
import org.apache.submarine.server.security.common.CommonConfig;

public class OidcConfig extends CommonConfig {

  private static final String CLIENT_ID_ENV = "SUBMARINE_AUTH_OIDC_CLIENT_ID";
  private static final String CLIENT_ID_PROP = "submarine.auth.oidc.client.id";

  private static final String CLIENT_SECRET_ENV = "SUBMARINE_AUTH_OIDC_CLIENT_SECRET";
  private static final String CLIENT_SECRET_PROP = "submarine.auth.oidc.client.secret";

  private static final String DISCOVER_URI_ENV = "SUBMARINE_AUTH_OIDC_DISCOVER_URI";
  private static final String DISCOVER_URI_PROP = "submarine.auth.oidc.discover.uri";

  private static final String LOGOUT_REDIRECT_URI_ENV = "SUBMARINE_AUTH_OIDC_LOGOUT_URI";
  private static final String LOGOUT_REDIRECT_URI_PROP = "submarine.auth.oidc.logout.uri";

  public static final String CLIENT_ID;

  public static final String CLIENT_SECRET;

  public static final String DISCOVER_URI;

  public static final String LOGOUT_REDIRECT_URI;

  static  {
    SubmarineConfiguration conf = SubmarineConfiguration.getInstance();
    CLIENT_ID = conf.getString(CLIENT_ID_ENV, CLIENT_ID_PROP, "");
    CLIENT_SECRET = conf.getString(CLIENT_SECRET_ENV, CLIENT_SECRET_PROP, "");
    DISCOVER_URI = conf.getString(DISCOVER_URI_ENV, DISCOVER_URI_PROP, "");
    LOGOUT_REDIRECT_URI = conf.getString(LOGOUT_REDIRECT_URI_ENV, LOGOUT_REDIRECT_URI_PROP, "");
  }
}
