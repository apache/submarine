package org.zapodot.junit.ldap.internal.unboundid;

import com.unboundid.ldap.sdk.LDAPConnection;

/**
 * A interface that is added to the LDAPInterface proxy to allow delegation.
 *
 * This interface is part of the internal API and may thus be changed or removed without warning.
 */
public interface LDAPConnectionProxy {

    void setLdapConnection(final LDAPConnection ldapConnection);
}
