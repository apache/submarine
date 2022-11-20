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
import org.apache.submarine.server.security.SecurityFactory;
import org.apache.submarine.server.security.common.CommonFilter;
import org.apache.submarine.server.security.common.RegistryUserActionAdapter;
import org.pac4j.core.config.Config;
import org.pac4j.core.context.JEEContext;
import org.pac4j.oidc.profile.OidcProfile;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.servlet.Filter;
import javax.servlet.FilterChain;
import javax.servlet.ServletException;
import javax.servlet.ServletRequest;
import javax.servlet.ServletResponse;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.IOException;
import java.util.Optional;

public class OidcFilter extends CommonFilter implements Filter {

  private static final Logger LOG = LoggerFactory.getLogger(OidcFilter.class);

  private final OidcSecurityProvider provider;
  private final Config oidcConfig;
  private final RegistryUserActionAdapter userActionAdapter = new RegistryUserActionAdapter();

  public OidcFilter() {
    this.provider = SecurityFactory.getPac4jSecurityProvider();
    this.oidcConfig = provider.getConfig();
  }

  @Override
  public void doFilter(ServletRequest request, ServletResponse response, FilterChain chain)
      throws IOException, ServletException {
    HttpServletRequest httpServletRequest = (HttpServletRequest) request;
    HttpServletResponse httpServletResponse = (HttpServletResponse) response;
    if (OidcCallbackResource.SELF_URL.equals(httpServletRequest.getRequestURI())) {
      CALLBACK_LOGIC.perform(
          new JEEContext(httpServletRequest, httpServletResponse, SESSION_STORE),
          oidcConfig,
          userActionAdapter,
          "/",
          true, false, false, null);
    } else if (OidcConfig.LOGOUT_ENDPOINT.equals(httpServletRequest.getRequestURI())) {
      String redirectUrl = OidcConfig.LOGOUT_REDIRECT_URI;
      if (StringUtils.isBlank(redirectUrl)) {
        redirectUrl = httpServletRequest.getParameter("redirect_url");
      }
      LOGOUT_LOGIC.perform(
          new JEEContext(httpServletRequest, httpServletResponse, SESSION_STORE),
          oidcConfig,
          DEFAULT_HTTP_ACTION_ADAPTER,
          redirectUrl,
          "/", true, true, true);
    }  else {
      Optional<OidcProfile> profile = provider.perform(httpServletRequest, httpServletResponse);
      if (profile.isPresent()) {
        chain.doFilter(request, response);
      } else if (httpServletRequest.getHeader(OidcConfig.AUTH_HEADER) != null) {
        httpServletResponse.sendError(HttpServletResponse.SC_UNAUTHORIZED, "The token is not valid.");
      }
    }
  }

  @Override
  public void destroy() {

  }
}
