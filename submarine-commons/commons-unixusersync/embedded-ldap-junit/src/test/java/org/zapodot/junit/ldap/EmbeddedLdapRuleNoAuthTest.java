package org.zapodot.junit.ldap;

import org.junit.Rule;
import org.junit.Test;

import static org.junit.Assert.assertNotNull;

public class EmbeddedLdapRuleNoAuthTest {

    @Rule
    public EmbeddedLdapRule embeddedLdapRule = EmbeddedLdapRuleBuilder
            .newInstance()
            .usingBindCredentials(null)
            .usingDomainDsn("dc=zapodot,dc=org")
            .importingLdifs("example.ldif")
            .build();

    @Test
    public void testConnect() throws Exception {
        assertNotNull(embeddedLdapRule.dirContext().search("cn=Sondre Eikanger Kvalo,ou=people,dc=zapodot,dc=org", null));

    }
}