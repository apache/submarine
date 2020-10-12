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

import javax.crypto.spec.SecretKeySpec;
import javax.naming.AuthenticationException;
import javax.naming.Context;
import javax.naming.NamingException;
import javax.naming.directory.DirContext;
import javax.naming.directory.InitialDirContext;
import javax.ws.rs.Consumes;
import javax.ws.rs.FormParam;
import javax.ws.rs.POST;
import javax.ws.rs.Path;
import javax.ws.rs.Produces;
import javax.ws.rs.core.MediaType;
import javax.ws.rs.core.Response;

import io.jsonwebtoken.JwtBuilder;
import io.jsonwebtoken.Jwts;
import io.jsonwebtoken.SignatureAlgorithm;
import io.jsonwebtoken.impl.crypto.MacProvider;
import org.apache.submarine.server.ldap.LdapManager;


import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.security.Key;
import java.util.Base64;
import java.util.Date;
import java.util.Hashtable;


@Path(RestConstants.V1 + "/" + RestConstants.LDAP)
@Produces({MediaType.APPLICATION_JSON + "; " + RestConstants.CHARSET_UTF8})

public class LdapAuthenticateRestApi {
  private final LdapManager ldapManager = LdapManager.getInstance();

  @POST
  @Produces(MediaType.APPLICATION_JSON)
  @Consumes(MediaType.APPLICATION_FORM_URLENCODED)
  public  Response authenticateUser(@FormParam("userName") String userName,
                                    @FormParam("password") String password) {
    try {
      //Authenticate the user using the credentials provided
      authenticate(userName, password);

      //create token for user
      String token = issueToken(userName);

      return Response.ok(token).build();
    }
    catch (Exception e) {
      return Response.status(Response.Status.FORBIDDEN).build();
    }
  }

  private static final Logger LOG = LoggerFactory.getLogger(LdapAuthenticateRestApi.class);

  private void authenticate(String username, String password) throws Exception {
    DirContext ctx = null;
    Hashtable<String, String> HashEnv = new Hashtable<>();

    String loginId = "uid=" + username + ",dc=example,dc=com";

    HashEnv.put(Context.SECURITY_AUTHENTICATION, "simple");
    HashEnv.put(Context.SECURITY_PRINCIPAL, loginId);
    HashEnv.put(Context.SECURITY_CREDENTIALS, password);
    HashEnv.put(Context.INITIAL_CONTEXT_FACTORY, "com.sun.jndi.ldap.LdapCtxFactory");
    HashEnv.put("com.sun.jndi.ldap.connect.timeout", "3000");
    HashEnv.put(Context.PROVIDER_URL, "ldap://ldap.forumsys.com:389");

    try {
      ctx = new InitialDirContext(HashEnv);
    } catch (AuthenticationException e) {
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

  private static final Key secret = MacProvider.generateKey(SignatureAlgorithm.HS256);
  private static final byte[] secretBytes = secret.getEncoded();
  private static final String base64SecretBytes = Base64.getEncoder().encodeToString(secretBytes);

  private String issueToken(String userName){
    long ttlMillis = 600;

    return createJWT(userName, ttlMillis);
  }

  private String createJWT(String id, long ttlMillis){


    // The JWT signature algorithm we will be using to sign the token
    SignatureAlgorithm signatureAlgorithm = SignatureAlgorithm.HS256;

    long nowMillis = System.currentTimeMillis();
    Date now = new Date(nowMillis);

    // We will sign our JWT with our ApiKey secret
    Key signingKey = new SecretKeySpec(secretBytes, signatureAlgorithm.getJcaName());

    //set the JWT Claims
    JwtBuilder builder = Jwts.builder().setId(id)
            .setIssuedAt(now)
            .signWith(signatureAlgorithm, signingKey);

    // if it has been specified, let's add the expiration
    if (ttlMillis >= 0) {
      long expMillis = nowMillis + ttlMillis;
      Date exp = new Date(expMillis);
      builder.setExpiration(exp);
    }

    // Builds the JWT and serializes it to a compact, URL-safe string
    return builder.compact();
  }

}
