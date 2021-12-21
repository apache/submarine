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

import org.apache.submarine.server.security.SecurityProvider;
import org.apache.submarine.server.security.common.CommonFilter;
import org.apache.submarine.server.security.defaultlogin.DefaultLoginConfig;
import org.pac4j.core.config.Config;
import org.pac4j.core.context.JEEContext;
import org.pac4j.core.http.callback.NoParameterCallbackUrlResolver;
import org.pac4j.core.http.url.DefaultUrlResolver;
import org.pac4j.core.profile.ProfileManager;
import org.pac4j.core.profile.UserProfile;
import org.pac4j.http.client.direct.HeaderClient;
import org.pac4j.oidc.client.OidcClient;
import org.pac4j.oidc.config.OidcConfiguration;
import org.pac4j.oidc.credentials.authenticator.UserInfoOidcAuthenticator;
import org.pac4j.oidc.profile.OidcProfile;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.util.Collection;
import java.util.Optional;

public class Pac4jSecurityProvider implements SecurityProvider<Pac4jFilter, OidcProfile> {

  private static final Logger LOG = LoggerFactory.getLogger(Pac4jSecurityProvider.class);

  private Config pac4jConfig;

  @Override
  public Class<Pac4jFilter> getFilterClass() {
    return Pac4jFilter.class;
  }

  @Override
  public Config getConfig() {
    if (pac4jConfig != null) {
      return pac4jConfig;
    }

    OidcConfiguration oidcConf = new OidcConfiguration();
    oidcConf.setClientId(OIDCConfig.CLIENT_ID);
    oidcConf.setSecret(OIDCConfig.CLIENT_SECRET);
    oidcConf.setDiscoveryURI(OIDCConfig.DISCOVER_URI);
    oidcConf.setExpireSessionWithToken(true);
    oidcConf.setUseNonce(true);
    oidcConf.setReadTimeout(5000);
    oidcConf.setMaxAge(OIDCConfig.MAX_AGE);

    OidcClient oidcClient = new OidcClient(oidcConf);
    oidcClient.setUrlResolver(new DefaultUrlResolver(true));
    oidcClient.setCallbackUrlResolver(new NoParameterCallbackUrlResolver());

    UserInfoOidcAuthenticator authenticator = new UserInfoOidcAuthenticator(oidcConf);
    HeaderClient headerClient = new HeaderClient(OIDCConfig.AUTH_HEADER, "Bearer ", authenticator);

    this.pac4jConfig = new Config(Pac4jCallbackResource.SELF_URL, oidcClient, headerClient);
    return pac4jConfig;
  }

  @Override
  public String getClient(HttpServletRequest httpServletRequest) {
    String client;
    if (httpServletRequest.getHeader(DefaultLoginConfig.AUTH_HEADER) != null) {
      client = "HeaderClient"; // use token
    } else {
      client = "OidcClient"; // use pac4j session
    }
    return client;
  }

  public OidcProfile perform(HttpServletRequest hsRequest, HttpServletResponse hsResponse) {
    JEEContext context = new JEEContext(hsRequest, hsResponse, CommonFilter.SESSION_STORE);
    UserProfile profile = CommonFilter.SECURITY_LOGIC.perform(
        context,
        getConfig(),
        (JEEContext ctx, Collection<UserProfile> profiles, Object... parameters) -> {
          if (profiles.isEmpty()) {
            LOG.warn("No profiles found after OIDC auth.");
            return null;
          } else {
            UserProfile next = profiles.iterator().next();
            return next;
          }
        },
        CommonFilter.DEFAULT_HTTP_ACTION_ADAPTER,
        getClient(hsRequest), DEFAULT_AUTHORIZER, null, null);
    return (OidcProfile) profile;
  }

  @Override
  public Optional<OidcProfile> getProfile(HttpServletRequest hsRequest, HttpServletResponse hsResponse) {
    JEEContext context = new JEEContext(hsRequest, hsResponse, CommonFilter.SESSION_STORE);
    ProfileManager<OidcProfile> manager = new ProfileManager<>(context);
    return manager.get(true);
  }
}
