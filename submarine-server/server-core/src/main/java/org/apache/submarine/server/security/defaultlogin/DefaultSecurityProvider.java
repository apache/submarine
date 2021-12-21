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

package org.apache.submarine.server.security.defaultlogin;

import org.apache.submarine.server.security.SecurityProvider;
import org.apache.submarine.server.security.common.CommonFilter;
import org.apache.submarine.server.security.oidc.OIDCConfig;
import org.pac4j.core.config.Config;
import org.pac4j.core.context.JEEContext;
import org.pac4j.core.matching.matcher.PathMatcher;
import org.pac4j.core.profile.ProfileManager;
import org.pac4j.core.profile.UserProfile;
import org.pac4j.http.client.direct.CookieClient;
import org.pac4j.http.client.direct.HeaderClient;
import org.pac4j.jwt.profile.JwtProfile;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.util.Collection;
import java.util.Optional;

public class DefaultSecurityProvider implements SecurityProvider<DefaultFilter, JwtProfile> {

  private static final Logger LOG = LoggerFactory.getLogger(DefaultSecurityProvider.class);

  private Config pac4jConfig;

  @Override
  public Class<DefaultFilter> getFilterClass() {
    return DefaultFilter.class;
  }

  @Override
  public Config getConfig() {
    if (pac4jConfig != null) {
      return pac4jConfig;
    }

    DefaultLoginConfig defaultConfig = new DefaultLoginConfig();
    CookieClient cookieClient = new CookieClient(DefaultLoginConfig.COOKIE_NAME,
            defaultConfig.getJwtAuthenticator());
    HeaderClient headerClient = new HeaderClient(OIDCConfig.AUTH_HEADER, "Bearer ",
            defaultConfig.getJwtAuthenticator());

    Config pac4jConfig = new Config(cookieClient, headerClient);
    pac4jConfig.addMatcher("static", new PathMatcher().excludeRegex(
            "^/.*(\\.map|\\.js|\\.css|\\.ico|\\.svg|\\.png|\\.html|\\.htm)$"));
    pac4jConfig.addMatcher("api", new PathMatcher().excludeRegex("^/api/auth/login$"));

    this.pac4jConfig = pac4jConfig;

    return pac4jConfig;
  }

  @Override
  public String getClient(HttpServletRequest httpServletRequest) {
    String client;
    if (httpServletRequest.getHeader(DefaultLoginConfig.AUTH_HEADER) != null) {
      client = "HeaderClient";
    } else {
      client = "CookieClient";
    }
    return client;
  }

  @Override
  public JwtProfile perform(HttpServletRequest hsRequest, HttpServletResponse hsResponse) {
    JEEContext context = new JEEContext(hsRequest, hsResponse, CommonFilter.SESSION_STORE);
    UserProfile profile = CommonFilter.SECURITY_LOGIC.perform(
        context,
        pac4jConfig,
        (JEEContext ctx, Collection<UserProfile> profiles, Object... parameters) -> {
          if (profiles.isEmpty()) {
            LOG.warn("No profiles found after Default auth.");
            return null;
          } else {
            return profiles.iterator().next();
          }
        },
        CommonFilter.DEFAULT_HTTP_ACTION_ADAPTER,
        getClient(hsRequest), DEFAULT_AUTHORIZER, "static,api", null);
    return (JwtProfile) profile;
  }

  @Override
  public Optional<JwtProfile> getProfile(HttpServletRequest hsRequest, HttpServletResponse hsResponse) {
    JEEContext context = new JEEContext(hsRequest, hsResponse, CommonFilter.SESSION_STORE);
    ProfileManager<JwtProfile> manager = new ProfileManager<>(context);
    return manager.get(true);
  }
}
