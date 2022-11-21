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
import org.pac4j.core.config.Config;
import org.pac4j.core.context.WebContext;
import org.pac4j.core.context.session.SessionStore;
import org.pac4j.core.engine.DefaultSecurityLogic;
import org.pac4j.core.engine.SecurityLogic;
import org.pac4j.core.http.adapter.HttpActionAdapter;
import org.pac4j.core.profile.UserProfile;
import org.pac4j.core.util.FindBest;
import org.pac4j.http.client.direct.HeaderClient;
import org.pac4j.jee.context.JEEContextFactory;
import org.pac4j.jee.context.session.JEESessionStoreFactory;
import org.pac4j.jee.http.adapter.JEEHttpActionAdapter;
import org.pac4j.jwt.profile.JwtProfile;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.util.Collection;
import java.util.Optional;

public class SimpleSecurityProvider extends SecurityProvider<SimpleFilter, JwtProfile> {

  private static final Logger LOG = LoggerFactory.getLogger(SimpleSecurityProvider.class);

  @Override
  public Class<SimpleFilter> getFilterClass() {
    return SimpleFilter.class;
  }

  @Override
  public Config createConfig() {
    if (pac4jConfig != null) {
      return pac4jConfig;
    }
    // header client
    HeaderClient headerClient = new HeaderClient(CommonConfig.AUTH_HEADER, CommonConfig.BEARER_HEADER_PREFIX,
            SimpleLoginConfig.getJwtAuthenticator());
    return new Config(headerClient);
  }

  @Override
  public String getClient(HttpServletRequest httpServletRequest) {
    return "HeaderClient";
  }

  @Override
  public Optional<JwtProfile> perform(HttpServletRequest hsRequest, HttpServletResponse hsResponse) {
    final WebContext context = FindBest.webContextFactory(null, this.pac4jConfig,
            JEEContextFactory.INSTANCE).newContext(hsRequest, hsResponse);
    final SessionStore sessionStore = FindBest.sessionStoreFactory(null, this.pac4jConfig,
            JEESessionStoreFactory.INSTANCE).newSessionStore(hsRequest, hsResponse);
    final HttpActionAdapter adapter = FindBest.httpActionAdapter(null, this.pac4jConfig,
            JEEHttpActionAdapter.INSTANCE);
    final SecurityLogic securityLogic = FindBest.securityLogic(null, this.pac4jConfig,
            DefaultSecurityLogic.INSTANCE);
    UserProfile profile = (UserProfile) securityLogic.perform(
        context,
        sessionStore,
        getConfig(),
        (WebContext ctx, SessionStore store, Collection<UserProfile> profiles, Object... parameters) -> {
          if (profiles.isEmpty()) {
            LOG.warn("No profiles found with default auth.");
            return null;
          } else {
            return profiles.iterator().next();
          }
        },
        adapter,
        getClient(hsRequest), DEFAULT_AUTHORIZER, "securityheaders"
    );
    return Optional.ofNullable((JwtProfile) profile);
  }
}
