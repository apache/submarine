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

import org.apache.submarine.commons.utils.SubmarineConfiguration;
import org.apache.submarine.server.security.common.CommonConfig;
import org.pac4j.jwt.config.encryption.SecretEncryptionConfiguration;
import org.pac4j.jwt.config.signature.SecretSignatureConfiguration;
import org.pac4j.jwt.credentials.authenticator.JwtAuthenticator;
import org.pac4j.jwt.profile.JwtGenerator;
import org.pac4j.jwt.profile.JwtProfile;

public class DefaultLoginConfig extends CommonConfig {

  public static final String COOKIE_NAME = "submarine_cookie";

  private static final String SUBMARINE_SECRET_ENV = "SUBMARINE_AUTH_DEFAULT_SECRET";
  private static final String SUBMARINE_SECRET_PROP = "submarine.auth.default.secret";

  private static final String DEFAULT_SUBMARINE_SECRET = "SUBMARINE_SECRET_12345678901234567890";

  private static final String SUBMARINE_SECRET;

  private final JwtAuthenticator jwtAuthenticator;

  static {
    SubmarineConfiguration conf = SubmarineConfiguration.getInstance();
    SUBMARINE_SECRET = conf.getString(SUBMARINE_SECRET_ENV, SUBMARINE_SECRET_PROP, DEFAULT_SUBMARINE_SECRET);
  }

  public DefaultLoginConfig() {
    this.jwtAuthenticator = new JwtAuthenticator(
            new SecretSignatureConfiguration(SUBMARINE_SECRET),
            new SecretEncryptionConfiguration(SUBMARINE_SECRET));
  }

  public JwtAuthenticator getJwtAuthenticator() {
    return jwtAuthenticator;
  }

  public static JwtGenerator<JwtProfile> getJwtGenerator() {
    return new JwtGenerator<>(new SecretSignatureConfiguration(SUBMARINE_SECRET));
  }
}
