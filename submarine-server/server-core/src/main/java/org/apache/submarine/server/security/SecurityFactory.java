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

package org.apache.submarine.server.security;

import org.apache.submarine.commons.utils.SubmarineConfVars;
import org.apache.submarine.commons.utils.SubmarineConfiguration;
import org.apache.submarine.server.security.oidc.Pac4jSecurityProvider;
import org.apache.submarine.server.security.defaultlogin.DefaultSecurityProvider;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashMap;
import java.util.Map;

public class SecurityFactory {

  private static final Logger LOG = LoggerFactory.getLogger(SecurityFactory.class);

  private static final String AUTH_TYPE;

  private static final Map<String, SecurityProvider> providerMap;

  public static Pac4jSecurityProvider getPac4jSecurityProvider() {
    return (Pac4jSecurityProvider) providerMap.get("oidc");
  }

  public static DefaultSecurityProvider getDefaultSecurityProvider() {
    return (DefaultSecurityProvider) providerMap.get("default");
  }

  static {
    SubmarineConfiguration conf = SubmarineConfiguration.getInstance();
    AUTH_TYPE = conf.getString(SubmarineConfVars.ConfVars.SUBMARINE_AUTH_TYPE);
    // int provider map
    providerMap = new HashMap<>();
    providerMap.put("oidc", new Pac4jSecurityProvider());
    providerMap.put("default", new DefaultSecurityProvider());
  }

  public static void addProvider(String name, SecurityProvider provider) {
    providerMap.put(name, provider);
  }

  public static SecurityProvider getSecurityProvider() {
    if (providerMap.containsKey(AUTH_TYPE)) {
      return providerMap.get(AUTH_TYPE);
    } else {
      LOG.warn("current auth type is {} but we can not recognize, so use default!", AUTH_TYPE);
      return getDefaultSecurityProvider();
    }
  }

}
