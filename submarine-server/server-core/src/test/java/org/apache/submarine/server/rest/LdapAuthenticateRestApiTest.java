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

package org.apache.submarine.server.rest;

import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.naming.AuthenticationException;
import javax.naming.Context;
import javax.naming.NamingException;
import javax.naming.directory.DirContext;
import javax.naming.directory.InitialDirContext;
import java.util.Hashtable;

import static org.junit.Assert.assertNotNull;

public class LdapAuthenticateRestApiTest {
  private static final Logger LOG = LoggerFactory.getLogger(LdapAuthenticateRestApiTest.class);

  @Test
  public void authenticate() {
    DirContext ctx = null;
    Hashtable<String, String> HashEnv = new Hashtable<>();

    String loginId = "uid=curie,dc=example,dc=com";
    String password = "password";

    HashEnv.put(Context.SECURITY_AUTHENTICATION, "simple");
    HashEnv.put(Context.SECURITY_PRINCIPAL, loginId);
    HashEnv.put(Context.SECURITY_CREDENTIALS, password);
    HashEnv.put(Context.INITIAL_CONTEXT_FACTORY, "com.sun.jndi.ldap.LdapCtxFactory");
    HashEnv.put("com.sun.jndi.ldap.connect.timeout", "3000");
    HashEnv.put(Context.PROVIDER_URL, "ldap://ldap.forumsys.com:389");

    try {
      ctx = new InitialDirContext(HashEnv);
      assertNotNull(ctx);
    } catch (AuthenticationException e) {
      LOG.error(e.getMessage(), e);
    } catch (javax.naming.CommunicationException e) {
      LOG.error(e.getMessage(), e);
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
    }

    if (ctx != null) {
      try {
        ctx.close();
      } catch (NamingException e) {
        LOG.error(e.getMessage(), e);
      }
    }
  }
}
