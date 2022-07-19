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

import org.apache.submarine.commons.utils.SubmarineConfiguration;
import org.apache.submarine.server.security.common.CommonConfig;
import org.pac4j.jwt.config.encryption.SecretEncryptionConfiguration;
import org.pac4j.jwt.config.signature.SecretSignatureConfiguration;
import org.pac4j.jwt.credentials.authenticator.JwtAuthenticator;
import org.pac4j.jwt.profile.JwtGenerator;

import static org.apache.submarine.commons.utils.SubmarineConfVars.ConfVars;

public class SimpleLoginConfig extends CommonConfig {

  private static final String SUBMARINE_SECRET;
  private static final JwtAuthenticator JWT_AUTHENTICATOR;
  private static final JwtGenerator JWT_GENERATOR;

  static {
    SubmarineConfiguration conf = SubmarineConfiguration.getInstance();
    // Generating the token requires a secret key,
    // if the user does not provide the secret key, we will use the default secret key
    SUBMARINE_SECRET = conf.getString(ConfVars.SUBMARINE_AUTH_DEFAULT_SECRET);
    JWT_AUTHENTICATOR = new JwtAuthenticator(
            new SecretSignatureConfiguration(SUBMARINE_SECRET),
            new SecretEncryptionConfiguration(SUBMARINE_SECRET));
    JWT_GENERATOR = new JwtGenerator(new SecretSignatureConfiguration(SUBMARINE_SECRET));
  }

  public static JwtAuthenticator getJwtAuthenticator() {
    return JWT_AUTHENTICATOR;
  }

  public static JwtGenerator getJwtGenerator() {
    return JWT_GENERATOR;
  }
}
