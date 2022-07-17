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

import org.pac4j.core.engine.DefaultCallbackLogic;
import org.pac4j.core.engine.DefaultLogoutLogic;
import org.pac4j.core.engine.DefaultSecurityLogic;
import org.pac4j.core.http.adapter.HttpActionAdapter;
import org.pac4j.jee.http.adapter.JEEHttpActionAdapter;

public class CommonFilter {

  public static final HttpActionAdapter DEFAULT_HTTP_ACTION_ADAPTER = JEEHttpActionAdapter.INSTANCE;

  public static final DefaultCallbackLogic CALLBACK_LOGIC =
          new DefaultCallbackLogic();

  public static final DefaultSecurityLogic SECURITY_LOGIC = new DefaultSecurityLogic();

  public static final DefaultLogoutLogic LOGOUT_LOGIC = new DefaultLogoutLogic();
}
