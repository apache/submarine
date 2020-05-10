package org.zapodot.junit.ldap;

import com.unboundid.ldap.sdk.LDAPConnection;
import com.unboundid.ldap.sdk.LDAPException;
import com.unboundid.ldap.sdk.LDAPInterface;
import org.junit.rules.TestRule;

import javax.naming.Context;
import javax.naming.NamingException;
import javax.naming.directory.DirContext;

/**
 * A JUnit rule that may be used as either a @Rule or a @ClassRule
 */
public interface EmbeddedLdapRule extends TestRule {

    /**
     * For tests depending on the UnboundID LDAP SDK. Returns a proxied version of an Unboundid interface that will be
     * closed when the test(s) have been invoked
     *
     * @return a shared LDAPConnection
     * @throws LDAPException if a connection can not be opened
     */
    LDAPInterface ldapConnection() throws LDAPException;

    /**
     * For tests depending on the UnboundID LDAP SDK that needs access to an ${link LDAPConnection} object
     * rather than the interface. If your code does not close the connection for you it will be closed on teardown
     *
     * @return a LDAPConnection connected to the embedded LDAP server
     * @throws LDAPException if an exception occurred while establishing the connection
     */
    LDAPConnection unsharedLdapConnection() throws LDAPException;

    /**
     * For tests depending on the standard Java JNDI API
     *
     * @return a shared Context connected to the in-memory LDAP server
     * @throws NamingException if context can not be created
     */
    Context context() throws NamingException;

    /**
     * Like {@link #context()}, but returns a DirContext
     *
     * @return a DirContext connected to the in-memory LDAP server
     * @throws NamingException if a LDAP failure happens during DirContext creation
     */
    DirContext dirContext() throws NamingException;

    /**
     * Gives access to the listening port for the currently running embedded LDAP server.
     * This will make it easier to use other integration mechanisms
     * <p>
     * Note: the embedded LDAP server is by default configured to listen only on the loopback address
     * (i.e <em>localhost/127.0.0.1</em>) unless another address has been provided to the builder when the rule was built
     * </p>
     *
     * @return the port number that the embedded server is listening to
     * @see org.zapodot.junit.ldap.EmbeddedLdapRuleBuilder#bindingToAddress(String)
     */
    int embeddedServerPort();


}
