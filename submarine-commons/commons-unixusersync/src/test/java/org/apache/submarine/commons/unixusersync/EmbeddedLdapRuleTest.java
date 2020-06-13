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
package org.apache.submarine.commons.unixusersync;

import com.google.common.collect.Iterators;

import com.unboundid.ldap.sdk.AddRequest;
import com.unboundid.ldap.sdk.Attribute;
import com.unboundid.ldap.sdk.LDAPConnection;
import com.unboundid.ldap.sdk.LDAPInterface;
import com.unboundid.ldap.sdk.SearchRequest;
import com.unboundid.ldap.sdk.SearchResult;
import com.unboundid.ldap.sdk.SearchResultEntry;
import com.unboundid.ldap.sdk.SearchScope;
import org.junit.Rule;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.zapodot.junit.ldap.EmbeddedLdapRule;
import org.zapodot.junit.ldap.EmbeddedLdapRuleBuilder;

import javax.naming.Context;
import javax.naming.NameClassPair;
import javax.naming.NamingEnumeration;
import javax.naming.directory.DirContext;
import javax.naming.directory.SearchControls;
import java.util.Arrays;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

public class EmbeddedLdapRuleTest {

  public static final String DOMAIN_DSN = "dc=zapodot,dc=org";
  private static final Logger LOG = LoggerFactory.getLogger(EmbeddedLdapRuleTest.class);

  @Rule
  public EmbeddedLdapRule embeddedLdapRule = EmbeddedLdapRuleBuilder
          .newInstance()
          .usingDomainDsn(DOMAIN_DSN)
          .importingLdifs("example.ldif")
          .build();

  @Test
  public void testLdapConnection() throws Exception {
    final LDAPInterface ldapConnection = embeddedLdapRule.ldapConnection();
    final SearchResult searchResult =
            ldapConnection.search(DOMAIN_DSN, SearchScope.SUB, "(objectClass=person)");

    assertEquals(2, searchResult.getEntryCount());
  }

  @Test
  public void testRawLdapConnection() throws Exception {
    final String commonName = "Test person";
    final String dn = String.format("cn=%s,ou=people,dc=zapodot,dc=org", commonName);
    LDAPConnection ldapConnection = embeddedLdapRule.unsharedLdapConnection();
    try {
      ldapConnection.add(new AddRequest(dn, Arrays.asList(
              new Attribute("objectclass", "top",
                      "person", "organizationalPerson", "inetOrgPerson"),
              new Attribute("cn", commonName), new Attribute("sn", "Person"),
              new Attribute("uid", "test"))));
    } finally {
      // Forces the LDAP connection to be closed.
      // This is not necessary as the rule will usually close it for you.
      ldapConnection.close();
    }
    ldapConnection = embeddedLdapRule.unsharedLdapConnection();
    final SearchResultEntry entry = ldapConnection.searchForEntry(new SearchRequest(dn,
          SearchScope.BASE,
          "(objectClass=person)"));
    assertNotNull(entry);
  }

  @Test
  public void testDirContext() throws Exception {
    final DirContext dirContext = embeddedLdapRule.dirContext();
    final SearchControls searchControls = new SearchControls();
    searchControls.setSearchScope(SearchControls.SUBTREE_SCOPE);
    final NamingEnumeration<javax.naming.directory.SearchResult> resultNamingEnumeration =
            dirContext.search(DOMAIN_DSN, "(objectClass=person)", searchControls);
    assertEquals(2, Iterators.size(Iterators.forEnumeration(resultNamingEnumeration)));
  }

  @Test
  public void testContext() throws Exception {
    final Context context = embeddedLdapRule.context();
    final Object user = context.lookup("cn=Eros,ou=people,dc=zapodot,dc=org");
    assertNotNull(user);
  }

  @Test
  public void testList() throws Exception {
    final Context context = embeddedLdapRule.context();
    NamingEnumeration list = context.list("ou=semi-people,dc=zapodot,dc=org");

    while (list.hasMore()){
      NameClassPair nc = (NameClassPair) list.next();
      String user = nc.getName().substring(3,nc.getName().length());
      LOG.info(user);

      assertEquals(user , "Fake-Eros");
    }

    context.close();
  }

  @Test
  public void testContextClose() throws Exception {
    final Context context = embeddedLdapRule.context();
    context.close();
    assertNotNull(context.getNameInNamespace());
  }

  @Test
  public void testEmbeddedServerPort() throws Exception {
    assertTrue(embeddedLdapRule.embeddedServerPort() > 0);
  }

  private void assertTrue(boolean b) {
  }

  @Test(expected = IllegalStateException.class)
  public void testNoPortAssignedYet() throws Exception {
    final EmbeddedLdapRule embeddedLdapRule = new EmbeddedLdapRuleBuilder().build();
    embeddedLdapRule.embeddedServerPort();
  }
}
