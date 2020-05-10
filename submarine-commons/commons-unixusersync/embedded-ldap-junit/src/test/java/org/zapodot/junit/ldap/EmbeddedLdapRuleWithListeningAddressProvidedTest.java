package org.zapodot.junit.ldap;

import org.junit.BeforeClass;
import org.junit.Rule;
import org.junit.Test;

import java.net.InetAddress;

import static org.junit.Assert.assertEquals;

public class EmbeddedLdapRuleWithListeningAddressProvidedTest {

    public static InetAddress inetAddress;

    @Rule
    public EmbeddedLdapRule embeddedLdapRule = EmbeddedLdapRuleBuilder
            .newInstance()
            .usingDomainDsn("dc=zapodot,dc=org")
            .importingLdifs("example.ldif")
            .bindingToAddress(inetAddress.getHostAddress())
            .build();

    @BeforeClass
    public static void setupAddress() throws Exception {
        inetAddress = InetAddress.getLocalHost();
    }

    @Test
    public void testLookupAddress() throws Exception {
        assertEquals(inetAddress.getHostAddress(),
                     embeddedLdapRule.unsharedLdapConnection().getConnectedAddress());

    }
}