package org.zapodot.junit.ldap.internal;

import com.google.common.base.Charsets;
import com.google.common.io.Resources;
import com.unboundid.ldap.listener.InMemoryDirectoryServer;
import com.unboundid.ldap.listener.InMemoryDirectoryServerConfig;
import com.unboundid.ldap.sdk.LDAPConnection;
import com.unboundid.ldap.sdk.LDAPException;
import com.unboundid.ldap.sdk.LDAPInterface;
import org.junit.runner.Description;
import org.junit.runners.model.Statement;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.zapodot.junit.ldap.EmbeddedLdapRule;
import org.zapodot.junit.ldap.internal.jndi.ContextProxyFactory;
import org.zapodot.junit.ldap.internal.unboundid.LDAPInterfaceProxyFactory;

import javax.naming.Context;
import javax.naming.NamingException;
import javax.naming.directory.DirContext;
import javax.naming.directory.InitialDirContext;
import javax.naming.ldap.LdapContext;
import java.io.UnsupportedEncodingException;
import java.net.URLDecoder;
import java.util.Hashtable;
import java.util.List;

public class EmbeddedLdapRuleImpl implements EmbeddedLdapRule {

    private static final String JAVA_RT_CONTROL_FACTORY = "com.sun.jndi.ldap.DefaultResponseControlFactory";

    private static final String JAVA_RT_CONTEXT_FACTORY = "com.sun.jndi.ldap.LdapCtxFactory";

    private static Logger logger = LoggerFactory.getLogger(EmbeddedLdapRuleImpl.class);
    private final InMemoryDirectoryServer inMemoryDirectoryServer;
    private final AuthenticationConfiguration authenticationConfiguration;
    private LDAPConnection ldapConnection;
    private InitialDirContext initialDirContext;
    private boolean isStarted = false;


    private EmbeddedLdapRuleImpl(final InMemoryDirectoryServer inMemoryDirectoryServer,
                                 final AuthenticationConfiguration authenticationConfiguration1) {
        this.inMemoryDirectoryServer = inMemoryDirectoryServer;
        this.authenticationConfiguration = authenticationConfiguration1;
    }

    public static EmbeddedLdapRule createForConfiguration(final InMemoryDirectoryServerConfig inMemoryDirectoryServerConfig,
                                                          final AuthenticationConfiguration authenticationConfiguration,
                                                          final List<String> ldifs) {
        try {
            return new EmbeddedLdapRuleImpl(createServer(inMemoryDirectoryServerConfig, ldifs),
                                            authenticationConfiguration);
        } catch (LDAPException e) {
            throw new IllegalStateException("Can not initiate in-memory LDAP server due to an exception", e);
        }
    }

    private static InMemoryDirectoryServer createServer(final InMemoryDirectoryServerConfig inMemoryDirectoryServerConfig,
                                                        final List<String> ldifs) throws LDAPException {
        final InMemoryDirectoryServer ldapServer =
                new InMemoryDirectoryServer(inMemoryDirectoryServerConfig);
        if (ldifs != null && !ldifs.isEmpty()) {
            for (final String ldif : ldifs) {
                try {
                    ldapServer.importFromLDIF(false, URLDecoder.decode(Resources.getResource(ldif).getPath(),
                                                                       Charsets.UTF_8.name()));
                } catch (UnsupportedEncodingException e) {
                    throw new IllegalStateException("Can not URL decode path:" + Resources.getResource(ldif).getPath(),
                                                    e);
                }
            }
        }
        return ldapServer;
    }

    @Override
    public LDAPInterface ldapConnection() throws LDAPException {
        return LDAPInterfaceProxyFactory.createProxy(createOrGetLdapConnection());
    }

    @Override
    public LDAPConnection unsharedLdapConnection() throws LDAPException {
        return createOrGetLdapConnection();
    }

    private LDAPConnection createOrGetLdapConnection() throws LDAPException {
        if (isStarted) {
            if (ldapConnection == null || ! ldapConnection.isConnected()) {
                ldapConnection = inMemoryDirectoryServer.getConnection();
            }
            return ldapConnection;
        } else {
            throw new IllegalStateException(
                    "Can not get a LdapConnection before the embedded LDAP server has been started");
        }
    }

    @Override
    public Context context() throws NamingException {
        return ContextProxyFactory.asDelegatingContext(createOrGetInitialDirContext());
    }

    @Override
    public DirContext dirContext() throws NamingException {
        return ContextProxyFactory.asDelegatingDirContext(createOrGetInitialDirContext());
    }

    @Override
    public int embeddedServerPort() {
        if(isStarted) {
            return inMemoryDirectoryServer.getListenPort();
        } else {
            throw new IllegalStateException("The embedded server must be started prior to accessing the listening port");
        }
    }

    private InitialDirContext createOrGetInitialDirContext() throws NamingException {
        if (isStarted) {
            if (initialDirContext == null) {
                initialDirContext = new InitialDirContext(createLdapEnvironment());
            }
            return initialDirContext;
        } else {
            throw new IllegalStateException(
                    "Can not get an InitialDirContext before the embedded LDAP server has been started");
        }
    }

    private Hashtable<String, String> createLdapEnvironment() {
        final Hashtable<String, String> environment = new Hashtable<>();
        environment.put(LdapContext.CONTROL_FACTORIES, JAVA_RT_CONTROL_FACTORY);
        environment.put(Context.PROVIDER_URL, String.format("ldap://%s:%s",
                                                            inMemoryDirectoryServer.getListenAddress().getHostName(),
                                                            embeddedServerPort()));
        environment.put(Context.INITIAL_CONTEXT_FACTORY, JAVA_RT_CONTEXT_FACTORY);
        if (authenticationConfiguration != null) {
            environment.putAll(authenticationConfiguration.toAuthenticationEnvironment());
        }
        return environment;
    }

    @Override
    public Statement apply(final Statement base, final Description description) {
        return statement(base);
    }

    private Statement statement(final Statement base) {
        return new Statement() {
            @Override
            public void evaluate() throws Throwable {
                startEmbeddedLdapServer();
                try {
                    base.evaluate();
                } finally {
                    takeDownEmbeddedLdapServer();
                }
            }
        };
    }

    private void startEmbeddedLdapServer() throws LDAPException {
        inMemoryDirectoryServer.startListening();
        isStarted = true;
    }

    private void takeDownEmbeddedLdapServer() {
        try {
            if (ldapConnection != null && ldapConnection.isConnected()) {
                ldapConnection.close();
            }
            if (initialDirContext != null) {
                initialDirContext.close();
            }
        } catch (NamingException e) {
            logger.info("Could not close initial context, forcing server shutdown anyway", e);
        } finally {
            inMemoryDirectoryServer.shutDown(true);
        }

    }


}
