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

package org.apache.submarine.server.security.simple;

import org.apache.submarine.server.security.SecurityProvider;
import org.apache.submarine.server.security.common.CommonConfig;
import org.apache.submarine.server.security.common.CommonFilter;
import org.pac4j.core.config.Config;
import org.pac4j.core.context.JEEContext;
import org.pac4j.core.matching.matcher.PathMatcher;
import org.pac4j.core.profile.ProfileManager;
import org.pac4j.core.profile.UserProfile;
import org.pac4j.http.client.direct.HeaderClient;
import org.pac4j.jwt.profile.JwtProfile;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.util.Collection;
import java.util.Optional;

public class SimpleSecurityProvider implements SecurityProvider<SimpleFilter, JwtProfile> {

  private static final Logger LOG = LoggerFactory.getLogger(SimpleSecurityProvider.class);

  private Config pac4jConfig;

  @Override
  public Class<SimpleFilter> getFilterClass() {
    return SimpleFilter.class;
  }

  @Override
  public Config getConfig() {
    if (pac4jConfig != null) {
      return pac4jConfig;
    }

    // header client
    HeaderClient headerClient = new HeaderClient(CommonConfig.AUTH_HEADER, CommonConfig.BEARER_HEADER_PREFIX,
            SimpleLoginConfig.getJwtAuthenticator());

    Config pac4jConfig = new Config(headerClient);
    // skip web static resources
    pac4jConfig.addMatcher("static", new PathMatcher().excludeRegex(
            "^/.*(\\.map|\\.js|\\.css|\\.ico|\\.svg|\\.png|\\.html|\\.htm)$"));
    // skip login rest api
    pac4jConfig.addMatcher("api", new PathMatcher().excludeRegex("^/api/auth/login$"));
    this.pac4jConfig = pac4jConfig;

    return pac4jConfig;
  }

  @Override
  public String getClient(HttpServletRequest httpServletRequest) {
    return "HeaderClient";
  }

  @Override
  public Optional<JwtProfile> perform(HttpServletRequest hsRequest, HttpServletResponse hsResponse) {
    JEEContext context = new JEEContext(hsRequest, hsResponse, CommonFilter.SESSION_STORE);
    UserProfile profile = CommonFilter.SECURITY_LOGIC.perform(
        context,
        pac4jConfig,
        (JEEContext ctx, Collection<UserProfile> profiles, Object... parameters) -> {
          if (profiles.isEmpty()) {
            LOG.warn("No profiles found with default auth.");
            return null;
          } else {
            return profiles.iterator().next();
          }
        },
        CommonFilter.DEFAULT_HTTP_ACTION_ADAPTER,
        getClient(hsRequest), DEFAULT_AUTHORIZER, "static,api", null);
    return Optional.ofNullable((JwtProfile) profile);
  }

  @Override
  public Optional<JwtProfile> getProfile(HttpServletRequest hsRequest, HttpServletResponse hsResponse) {
    JEEContext context = new JEEContext(hsRequest, hsResponse, CommonFilter.SESSION_STORE);
    ProfileManager<JwtProfile> manager = new ProfileManager<>(context);
    return manager.get(true);
  }
}
