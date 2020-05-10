package org.zapodot.junit.ldap.internal;

import org.junit.Test;

import java.util.Map;

import static org.junit.Assert.assertEquals;

public class AuthenticationConfigurationTest {

    @Test
    public void testToAuthenticationEnvironment() throws Exception {
        final AuthenticationConfiguration authenticationConfiguration = new AuthenticationConfiguration(
                "cn=someone,ou=people,dc=net",
                "credentials");
        final Map<String, String> environment = authenticationConfiguration.toAuthenticationEnvironment();
        assertEquals(3, environment.size());
    }

    @Test(expected = IllegalStateException.class)
    public void testNotSet() throws Exception {
        new AuthenticationConfiguration(null, null).toAuthenticationEnvironment();

    }
}