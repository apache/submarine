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

import org.apache.submarine.server.security.SecurityFactory;
import org.apache.submarine.server.security.common.CommonFilter;
import org.pac4j.jwt.profile.JwtProfile;
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

/**
 * Simple authentication
 * Only users in submarine sys_user table can log in, and user is verified based on token
 */
public class SimpleFilter extends CommonFilter implements Filter {

  private static final Logger LOG = LoggerFactory.getLogger(SimpleFilter.class);

  private final SimpleSecurityProvider provider;

  public SimpleFilter() {
    this.provider = SecurityFactory.getSimpleSecurityProvider();
  }

  @Override
  public void doFilter(ServletRequest servletRequest, ServletResponse servletResponse,
                       FilterChain filterChain) throws IOException, ServletException {
    HttpServletRequest httpServletRequest = (HttpServletRequest) servletRequest;
    HttpServletResponse httpServletResponse = (HttpServletResponse) servletResponse;
    if (isProtectedApi(httpServletRequest)) {
      // check header token
      Optional<JwtProfile> profile = provider.perform(httpServletRequest, httpServletResponse);
      // If the token can be correctly parsed then continue processing, otherwise return 401
      if (profile.isPresent()) {
        filterChain.doFilter(servletRequest, servletResponse);
      } else {
        httpServletResponse.sendError(HttpServletResponse.SC_UNAUTHORIZED, "The token is not valid.");
      }
    } else {
      filterChain.doFilter(servletRequest, servletResponse);
    }
  }

  @Override
  public void destroy() {
  }
}
