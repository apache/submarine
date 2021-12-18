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

import org.apache.submarine.server.security.SecurityFactory;
import org.apache.submarine.server.security.common.CommonFilter;
import org.apache.submarine.server.security.oidc.OIDCConfig;
import org.pac4j.core.config.Config;
import org.pac4j.core.context.HttpConstants;
import org.pac4j.core.context.JEEContext;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.servlet.Filter;
import javax.servlet.FilterChain;
import javax.servlet.FilterConfig;
import javax.servlet.ServletException;
import javax.servlet.ServletRequest;
import javax.servlet.ServletResponse;
import javax.servlet.http.Cookie;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.IOException;

public class DefaultFilter extends CommonFilter implements Filter {

  private static final Logger LOG = LoggerFactory.getLogger(DefaultFilter.class);

  private static final String USER_LOGIN = "/user/login";

  private DefaultSecurityProvider provider;

  private final Config pac4jConfig;

  public DefaultFilter() {
    this.provider = SecurityFactory.getDefaultSecurityProvider();
    this.pac4jConfig = provider.getConfig();
  }

  @Override
  public void init(FilterConfig filterConfig) throws ServletException {

  }

  @Override
  public void doFilter(ServletRequest servletRequest, ServletResponse servletResponse,
                       FilterChain filterChain) throws IOException, ServletException {

    HttpServletRequest httpServletRequest = (HttpServletRequest) servletRequest;
    HttpServletResponse httpServletResponse = (HttpServletResponse) servletResponse;
    JEEContext context = new JEEContext(httpServletRequest, httpServletResponse, SESSION_STORE);

    if (isUserAgent(httpServletRequest)) {
      filterChain.doFilter(servletRequest, servletResponse);
    } else if (OIDCConfig.LOGOUT_ENDPOINT.equals(httpServletRequest.getRequestURI())) {
      LOGOUT_LOGIC.perform(
          context,
          pac4jConfig,
          DEFAULT_HTTP_ACTION_ADAPTER,
          "/",
          "/", true, true, true);
      Cookie[] cookies = httpServletRequest.getCookies();
      if (cookies != null)
        for (Cookie cookie : cookies) {
          cookie.setValue("");
          cookie.setPath("/");
          cookie.setMaxAge(0);
          httpServletResponse.addCookie(cookie);
        }
      httpServletResponse.sendRedirect(USER_LOGIN);
    } else if (!USER_LOGIN.equals(httpServletRequest.getRequestURI())) {
      provider.perform(httpServletRequest, httpServletResponse);
      if (!"HeaderClient".equals(provider.getClient(httpServletRequest))
              && httpServletResponse.getStatus() == HttpConstants.UNAUTHORIZED) {
        ((org.eclipse.jetty.server.Response) httpServletResponse).reopen();
        httpServletResponse.sendRedirect(USER_LOGIN);
      } else {
        filterChain.doFilter(servletRequest, servletResponse);
      }
    } else {
      filterChain.doFilter(servletRequest, servletResponse);
    }
  }

  @Override
  public void destroy() {

  }
}
