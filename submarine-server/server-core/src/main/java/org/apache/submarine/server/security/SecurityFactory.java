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
import org.apache.submarine.server.security.oidc.OidcSecurityProvider;
import org.apache.submarine.server.security.simple.SimpleSecurityProvider;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashMap;
import java.util.Map;
import java.util.Optional;

public class SecurityFactory {

  private static final Logger LOG = LoggerFactory.getLogger(SecurityFactory.class);

  private static final Map<String, SecurityProvider> providerMap;
  public static final String AUTH_TYPE;

  public static SimpleSecurityProvider getSimpleSecurityProvider() {
    return (SimpleSecurityProvider) providerMap.get("simple");
  }

  public static OidcSecurityProvider getPac4jSecurityProvider() {
    return (OidcSecurityProvider) providerMap.get("oidc");
  }

  static {
    AUTH_TYPE = SubmarineConfiguration.getInstance()
            .getString(SubmarineConfVars.ConfVars.SUBMARINE_AUTH_TYPE);
    // int provider map
    providerMap = new HashMap<>();
    providerMap.put("simple", new SimpleSecurityProvider());
    providerMap.put("oidc", new OidcSecurityProvider());
  }

  public static void addProvider(String name, SecurityProvider provider) {
    providerMap.put(name, provider);
  }

  public static Optional<SecurityProvider> getSecurityProvider() {
    if (providerMap.containsKey(AUTH_TYPE)) {
      return Optional.ofNullable(providerMap.get(AUTH_TYPE));
    } else {
      LOG.warn("current auth type is {} but we can not recognize, so use none!", AUTH_TYPE);
      return Optional.empty();
    }
  }

}
