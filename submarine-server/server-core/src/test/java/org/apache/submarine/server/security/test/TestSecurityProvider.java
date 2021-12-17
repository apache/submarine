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

package org.apache.submarine.server.security.test;

import org.apache.submarine.server.security.SecurityProvider;
import org.pac4j.core.config.Config;
import org.pac4j.core.profile.AnonymousProfile;

import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.util.Optional;

public class TestSecurityProvider implements SecurityProvider<TestFilter, AnonymousProfile> {

  @Override
  public Class<TestFilter> getFilterClass() {
    return TestFilter.class;
  }

  @Override
  public Config getConfig() {
    return null;
  }

  @Override
  public String getClient(HttpServletRequest httpServletRequest) {
    return null;
  }

  @Override
  public AnonymousProfile perform(HttpServletRequest hsRequest, HttpServletResponse hsResponse) {
    return new AnonymousProfile();
  }

  @Override
  public Optional<AnonymousProfile> getProfile(HttpServletRequest hsRequest,
                                               HttpServletResponse hsResponse) {
    return Optional.of(new AnonymousProfile());
  }
}
