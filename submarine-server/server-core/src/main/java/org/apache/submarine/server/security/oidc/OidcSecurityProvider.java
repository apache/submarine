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

import org.apache.commons.lang3.StringUtils;
import org.apache.submarine.commons.utils.SubmarineConfVars;
import org.apache.submarine.commons.utils.SubmarineConfiguration;
import org.apache.submarine.server.security.SecurityProvider;
import org.apache.submarine.server.security.common.AuthFlowType;
import org.apache.submarine.server.security.common.RegistryUserActionAdapter;
import org.pac4j.core.authorization.authorizer.Authorizer;
import org.pac4j.core.config.Config;
import org.pac4j.core.context.WebContext;
import org.pac4j.core.context.session.SessionStore;
import org.pac4j.core.engine.CallbackLogic;
import org.pac4j.core.engine.DefaultCallbackLogic;
import org.pac4j.core.engine.DefaultLogoutLogic;
import org.pac4j.core.engine.DefaultSecurityLogic;
import org.pac4j.core.engine.LogoutLogic;
import org.pac4j.core.engine.SecurityLogic;
import org.pac4j.core.http.adapter.HttpActionAdapter;
import org.pac4j.core.http.callback.NoParameterCallbackUrlResolver;
import org.pac4j.core.http.url.DefaultUrlResolver;
import org.pac4j.core.matching.matcher.Matcher;
import org.pac4j.core.matching.matcher.csrf.CsrfTokenGeneratorMatcher;
import org.pac4j.core.matching.matcher.csrf.DefaultCsrfTokenGenerator;
import org.pac4j.core.profile.UserProfile;
import org.pac4j.core.util.FindBest;
import org.pac4j.http.client.direct.HeaderClient;
import org.pac4j.jee.context.JEEContextFactory;
import org.pac4j.jee.context.session.JEESessionStoreFactory;
import org.pac4j.jee.http.adapter.JEEHttpActionAdapter;
import org.pac4j.oidc.client.OidcClient;
import org.pac4j.oidc.config.OidcConfiguration;
import org.pac4j.oidc.credentials.authenticator.UserInfoOidcAuthenticator;
import org.pac4j.oidc.profile.OidcProfile;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;

import static org.pac4j.core.matching.matcher.DefaultMatchers.CSRF_TOKEN;

public class OidcSecurityProvider extends SecurityProvider<OidcFilter, OidcProfile> {

  private static final Logger LOG = LoggerFactory.getLogger(OidcSecurityProvider.class);

  private final RegistryUserActionAdapter userActionAdapter = new RegistryUserActionAdapter();

  @Override
  public AuthFlowType getAuthFlowType() {
    return AuthFlowType.SESSION;
  }


  @Override
  public Class<OidcFilter> getFilterClass() {
    return OidcFilter.class;
  }

  @Override
  public Config createConfig() {
    // oidc config
    OidcConfiguration oidcConf = new OidcConfiguration();
    oidcConf.setClientId(OidcConfig.CLIENT_ID);
    oidcConf.setSecret(OidcConfig.CLIENT_SECRET);
    oidcConf.setDiscoveryURI(OidcConfig.DISCOVER_URI);
    oidcConf.setExpireSessionWithToken(true);
    oidcConf.setUseNonce(true);
    oidcConf.setReadTimeout(5000);
    oidcConf.setMaxAge(OidcConfig.MAX_AGE);
    // oidc client
    OidcClient oidcClient = new OidcClient(oidcConf);
    oidcClient.setUrlResolver(new DefaultUrlResolver(true));
    oidcClient.setCallbackUrlResolver(new NoParameterCallbackUrlResolver());
    // header client
    HeaderClient headerClient = new HeaderClient(OidcConfig.AUTH_HEADER,
            OidcConfig.BEARER_HEADER_PREFIX, new UserInfoOidcAuthenticator(oidcConf));
    Config config = new Config(OidcCallbackResource.SELF_URL, oidcClient, headerClient);
    // add csrfToken matcher
    SubmarineConfiguration conf = SubmarineConfiguration.getInstance();
    CsrfTokenGeneratorMatcher csrftgm = new CsrfTokenGeneratorMatcher(new DefaultCsrfTokenGenerator());
    csrftgm.setSecure(conf.getBoolean(SubmarineConfVars.ConfVars.SUBMARINE_COOKIE_SECURE));
    csrftgm.setHttpOnly(conf.getBoolean(SubmarineConfVars.ConfVars.SUBMARINE_COOKIE_HTTP_ONLY));
    config.setMatchers(Map.of(CSRF_TOKEN, csrftgm));
    return config;
  }

  @Override
  public String getClient(HttpServletRequest httpServletRequest) {
    return "OidcClient,HeaderClient";
  }

  @Override
  public Optional<OidcProfile> perform(HttpServletRequest hsRequest, HttpServletResponse hsResponse) {
    final WebContext context = FindBest.webContextFactory(null, this.pac4jConfig,
            JEEContextFactory.INSTANCE).newContext(hsRequest, hsResponse);
    final SessionStore sessionStore = FindBest.sessionStoreFactory(null, this.pac4jConfig,
            JEESessionStoreFactory.INSTANCE).newSessionStore(hsRequest, hsResponse);
    final HttpActionAdapter adapter = FindBest.httpActionAdapter(null, this.pac4jConfig,
            JEEHttpActionAdapter.INSTANCE);
    final SecurityLogic securityLogic = FindBest.securityLogic(null, this.pac4jConfig,
            DefaultSecurityLogic.INSTANCE);
    // perform get profile
    UserProfile profile = (UserProfile) securityLogic.perform(
        context,
        sessionStore,
        getConfig(),
        (WebContext ctx, SessionStore store, Collection<UserProfile> profiles, Object... parameters) -> {
          if (profiles.isEmpty()) {
            LOG.warn("No profiles found after OIDC auth.");
            return null;
          } else {
            return profiles.iterator().next();
          }
        },
        adapter,
        getClient(hsRequest), DEFAULT_AUTHORIZER, ""
    );
    return Optional.ofNullable((OidcProfile) profile);
  }

  @Override
  public void callback(HttpServletRequest hsRequest, HttpServletResponse hsResponse) {
    final CallbackLogic bestLogic = FindBest.callbackLogic(null, this.pac4jConfig,
        DefaultCallbackLogic.INSTANCE);
    final WebContext context = FindBest.webContextFactory(null, this.pac4jConfig,
        JEEContextFactory.INSTANCE).newContext(hsRequest, hsResponse);
    final SessionStore sessionStore = FindBest.sessionStoreFactory(null, this.pac4jConfig,
        JEESessionStoreFactory.INSTANCE).newSessionStore(hsRequest, hsResponse);
    // perform callback
    bestLogic.perform(context, sessionStore, this.pac4jConfig, userActionAdapter, "/",
        false, "oidcClient");
  }

  @Override
  public void logout(HttpServletRequest hsRequest, HttpServletResponse hsResponse) {
    String redirectUrl = OidcConfig.LOGOUT_REDIRECT_URI;
    if (StringUtils.isBlank(redirectUrl)) {
      redirectUrl = hsRequest.getParameter("redirect_url");
    }
    final HttpActionAdapter bestAdapter = FindBest.httpActionAdapter(null, this.pac4jConfig,
        JEEHttpActionAdapter.INSTANCE);
    final LogoutLogic bestLogic = FindBest.logoutLogic(null, this.pac4jConfig,
        DefaultLogoutLogic.INSTANCE);
    final WebContext context = FindBest.webContextFactory(null, this.pac4jConfig,
        JEEContextFactory.INSTANCE).newContext(hsRequest, hsResponse);
    final SessionStore sessionStore = FindBest.sessionStoreFactory(null, this.pac4jConfig,
        JEESessionStoreFactory.INSTANCE).newSessionStore(hsRequest, hsResponse);
    // perform logout
    bestLogic.perform(context, sessionStore, this.pac4jConfig, bestAdapter, redirectUrl, "/",
        true, true, true);
  }

}
