package org.zapodot.junit.ldap;

import com.unboundid.ldap.sdk.LDAPInterface;
import org.junit.Rule;
import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;

public class EmbeddedLdapRuleMultipleDSNs {

    public static final String DSN_ROOT_ONE = "dc=zapodot,dc=com";
    public static final String DSN_ROOT_TWO = "dc=zapodot,dc=org";
    @Rule
    public EmbeddedLdapRule embeddedLdapRule = EmbeddedLdapRuleBuilder.newInstance()
                                                                      .usingDomainDsn(DSN_ROOT_ONE)
                                                                      .usingDomainDsn(DSN_ROOT_TWO)
                                                                      .importingLdifs("example.ldif")
                                                                      .build();

    @Test
    public void testCheckNamingContexts() throws Exception {
        final LDAPInterface ldapConnection = embeddedLdapRule.ldapConnection();
        final String[] namingContextDNs = ldapConnection.getRootDSE().getNamingContextDNs();
        assertArrayEquals(new String[]{DSN_ROOT_ONE, DSN_ROOT_TWO}, namingContextDNs);

    }
}
