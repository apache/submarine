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

public class OidcSecurityProvider extends SecurityProvider<OidcFilter, OidcProfile> {

  private static final Logger LOG = LoggerFactory.getLogger(OidcSecurityProvider.class);

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
    return new Config(OidcCallbackResource.SELF_URL, oidcClient, headerClient);
  }

  @Override
  public String getClient(HttpServletRequest httpServletRequest) {
    return "OidcClient,HeaderClient";
  }

  @Override
  public Optional<OidcProfile> perform(HttpServletRequest hsRequest, HttpServletResponse hsResponse) {
    JEEContext context = new JEEContext(hsRequest, hsResponse, CommonFilter.SESSION_STORE);
    UserProfile profile = CommonFilter.SECURITY_LOGIC.perform(
        context,
        getConfig(),
        (JEEContext ctx, Collection<UserProfile> profiles, Object... parameters) -> {
          if (profiles.isEmpty()) {
            LOG.warn("No profiles found after OIDC auth.");
            return null;
          } else {
            return profiles.iterator().next();
          }
        },
        CommonFilter.DEFAULT_HTTP_ACTION_ADAPTER,
        getClient(hsRequest), DEFAULT_AUTHORIZER, null, null);
    return Optional.ofNullable((OidcProfile) profile);
  }

}
