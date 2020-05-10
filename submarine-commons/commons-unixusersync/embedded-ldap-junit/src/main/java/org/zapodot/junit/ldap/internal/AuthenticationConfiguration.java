package org.zapodot.junit.ldap.internal;

import com.google.common.collect.ImmutableMap;

import javax.naming.Context;
import java.util.Map;

/**
 * LDAP authentication POJO.
 *
 * This class is part of the internal API and may thus be changed or removed without warning.
 */
public class AuthenticationConfiguration {
    public final String userDn;
    public final String credentials;

    public AuthenticationConfiguration(final String userDn, final String credentials) {
        this.userDn = userDn;
        this.credentials = credentials;
    }


    public Map<String, String> toAuthenticationEnvironment() {

        if (userDn == null || credentials == null) {
            throw new IllegalStateException("userDn and credentials must be set before generating the "
                                               + "authentication environment");
        }

        return ImmutableMap.<String, String>builder()
                           .put(Context.SECURITY_PRINCIPAL, userDn)
                           .put(Context.SECURITY_PROTOCOL, "simple")
                           .put(Context.SECURITY_CREDENTIALS, credentials)
                           .build();
    }
}
