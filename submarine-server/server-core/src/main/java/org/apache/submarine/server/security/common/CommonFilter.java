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

package org.apache.submarine.server.security.common;

import org.apache.commons.lang3.StringUtils;
import org.pac4j.core.context.JEEContext;
import org.pac4j.core.context.session.JEESessionStore;
import org.pac4j.core.context.session.SessionStore;
import org.pac4j.core.engine.DefaultCallbackLogic;
import org.pac4j.core.engine.DefaultLogoutLogic;
import org.pac4j.core.engine.DefaultSecurityLogic;
import org.pac4j.core.http.adapter.HttpActionAdapter;
import org.pac4j.core.http.adapter.JEEHttpActionAdapter;
import org.pac4j.core.profile.CommonProfile;
import org.pac4j.core.profile.UserProfile;

import javax.servlet.http.HttpServletRequest;

public class CommonFilter {

  public static final HttpActionAdapter DEFAULT_HTTP_ACTION_ADAPTER = JEEHttpActionAdapter.INSTANCE;

  public static final DefaultCallbackLogic<CommonProfile, JEEContext> CALLBACK_LOGIC =
      new DefaultCallbackLogic<>();

  public static final DefaultSecurityLogic<UserProfile, JEEContext> SECURITY_LOGIC =
      new DefaultSecurityLogic<>();

  public static final DefaultLogoutLogic<UserProfile, JEEContext> LOGOUT_LOGIC = new DefaultLogoutLogic<>();

  public static final SessionStore<JEEContext> SESSION_STORE = new JEESessionStore();

  /**
   * If it is called by python, temporarily passed
   */
  protected boolean isUserAgent(HttpServletRequest httpServletRequest) {
    String agentHeader = httpServletRequest.getHeader(CommonConfig.AGENT_HEADER);
    if (StringUtils.isNoneBlank(agentHeader)) return CommonConfig.PYTHON_USER_AGENT.equals(agentHeader);
    return false;
  }

}
