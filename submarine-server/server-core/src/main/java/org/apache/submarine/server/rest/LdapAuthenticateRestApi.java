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

import org.apache.submarine.server.ldap.LdapManager;

import java.util.Hashtable;


@Path(RestConstants.V1 + "/" + RestConstants.LDAP)
@Produces({MediaType.APPLICATION_JSON + "; " + RestConstants.CHARSET_UTF8})

public class LdapAuthenticateRestApi {
  private final LdapManager ldapManager = LdapManager.getInstance();

  @POST
  @Produces(MediaType.APPLICATION_JSON)
  @Consumes(MediaType.APPLICATION_FORM_URLENCODED)
  public  Response authenticateUser(@FormParam("username") String username,
                                    @FormParam("password") String password) {
    try {
      //Authenticate the user using the credentials provided
      authenticate(username, password);

      return Response.ok().build();
    }
    catch (Exception e) {
      return Response.status(Response.Status.FORBIDDEN).build();
    }
  }

  private  void authenticate(String username, String password) throws Exception {
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
    }
    catch (AuthenticationException e) {
      e.printStackTrace();
    }

    if (ctx != null) {
      try {
        ctx.close();
      }
      catch (NamingException e) {
        e.printStackTrace();
      }
    }
  }

}
