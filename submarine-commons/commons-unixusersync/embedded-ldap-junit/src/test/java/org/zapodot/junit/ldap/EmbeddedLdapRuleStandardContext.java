package org.zapodot.junit.ldap;

import org.junit.Rule;
import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;

public class EmbeddedLdapRuleStandardContext {

    @Rule
    public EmbeddedLdapRule embeddedLdapRule = EmbeddedLdapRuleBuilder.newInstance()
                                                                      .build();

    @Test
    public void testUsingDefaultDomain() throws Exception {
        assertArrayEquals(new String[]{EmbeddedLdapRuleBuilder.DEFAULT_DOMAIN},
                          embeddedLdapRule.ldapConnection().getRootDSE().getNamingContextDNs());


    }
}
