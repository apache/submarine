package org.zapodot.junit.ldap;

import org.junit.Rule;
import org.junit.Test;

import static org.junit.Assert.assertNotNull;

public class EmbeddedLdapRuleWithSpacesTest {

    @Rule
    public EmbeddedLdapRule embeddedLdapRule = EmbeddedLdapRuleBuilder
            .newInstance()
            .usingDomainDsn("dc=zapodot,dc=org")
            .importingLdifs("folder with space/example.ldif")
            .build();

    @Test
    public void testIsUp() throws Exception {
        assertNotNull(embeddedLdapRule.ldapConnection().getRootDSE());

    }
}
