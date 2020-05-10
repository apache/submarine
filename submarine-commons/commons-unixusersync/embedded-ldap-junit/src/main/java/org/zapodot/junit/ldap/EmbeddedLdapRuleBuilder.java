package org.zapodot.junit.ldap;

import com.google.common.base.Function;
import com.google.common.base.Optional;
import com.google.common.collect.Lists;
import com.google.common.io.Resources;
import com.unboundid.ldap.listener.InMemoryDirectoryServerConfig;
import com.unboundid.ldap.listener.InMemoryListenerConfig;
import com.unboundid.ldap.sdk.LDAPException;
import com.unboundid.ldap.sdk.schema.Schema;
import com.unboundid.ldif.LDIFException;
import org.zapodot.junit.ldap.internal.AuthenticationConfiguration;
import org.zapodot.junit.ldap.internal.EmbeddedLdapRuleImpl;

import java.io.File;
import java.io.IOException;
import java.net.InetAddress;
import java.net.URISyntaxException;
import java.net.UnknownHostException;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.Objects;

/**
 * A builder providing a fluent way of defining EmbeddedLdapRule instances
 */
public class EmbeddedLdapRuleBuilder {

    public static final String DEFAULT_DOMAIN = "dc=example,dc=com";
    public static final String DEFAULT_BIND_DSN = "cn=Directory manager";
    public static final String DEFAULT_BIND_CREDENTIALS = "password";
    public static final String LDAP_SERVER_LISTENER_NAME = "test-listener";
    public static final int MIN_PORT_EXCLUSIVE = 0;
    public static final int MAX_PORT_EXCLUSIVE = 65535;
    private List<String> domainDsn = new LinkedList<>();

    private String bindDSN = DEFAULT_BIND_DSN;

    private String bindCredentials = DEFAULT_BIND_CREDENTIALS;

    private List<String> ldifsToImport = new LinkedList<>();

    private List<String> schemaLdifs = new LinkedList<>();

    private boolean addDefaultSchema = true;

    private Integer bindPort = 0;

    private InetAddress bindAddress = InetAddress.getLoopbackAddress();

    private AuthenticationConfiguration authenticationConfiguration;

    public EmbeddedLdapRuleBuilder() {
    }

    /**
     * Creates a new builder
     *
     * @return a new EmbeddedLdapRuleBuilder instance
     */
    public static EmbeddedLdapRuleBuilder newInstance() {
        return new EmbeddedLdapRuleBuilder();
    }

    /**
     * Sets a domainDsn to be used. May be multiple values. If not set, it will default to the value of the {@link #DEFAULT_DOMAIN DEFAULT_DOMAIN} field
     *
     * @param domainDsn a valid DSN string
     * @return same EmbeddedLdapRuleBuilder instance with the domainDsn field set
     */
    public EmbeddedLdapRuleBuilder usingDomainDsn(final String domainDsn) {
        this.domainDsn.add(domainDsn);
        return this;
    }

    /**
     * Sets the DSN to bind to when authenticating. If not set, it will default to the value of the {@link #DEFAULT_BIND_DSN DEFAULT_BIND_DSN} field
     *
     * @param bindDSN a valid DSN string
     * @return same EmbeddedLdapRuleBuilder instance with the bindDSN field set
     */
    public EmbeddedLdapRuleBuilder usingBindDSN(final String bindDSN) {
        this.bindDSN = bindDSN;
        return this;
    }

    /**
     * Sets the credentials to be used to authenticate. If not set, it will default to the value of the {@link #DEFAULT_BIND_CREDENTIALS DEFAULT_BIND_CREDENTIALS} field
     *
     * @param bindCredentials a password string
     * @return same EmbeddedLdapRuleBuilder instance with the bindCredentials field set
     */
    public EmbeddedLdapRuleBuilder usingBindCredentials(final String bindCredentials) {
        this.bindCredentials = bindCredentials;
        return this;
    }

    /**
     * Sets the port that the in-memory LDAP server will bind to. If not set, an available port will be picked automatically
     *
     * @param port a port number
     * @return same EmbeddedLdapRuleBuilder instance with the port field set
     * @throws IllegalArgumentException if the provided value for port is not between @{link MIN_PORT_EXCLUSIVE}
     *                                  and @{MAX_PORT_EXCLUSIVE} (exclusive)
     */
    public EmbeddedLdapRuleBuilder bindingToPort(final int port) {
        if ((port < MIN_PORT_EXCLUSIVE) || (port > MAX_PORT_EXCLUSIVE)) {
            throw new IllegalArgumentException(String.format("Value \"%s\" is not a valid port number", port));
        }
        this.bindPort = Integer.valueOf(port);
        return this;
    }

    /**
     * Allows the listening address for the embedded LDAP server to be set. If not set it will bind to <em>localhost/127.0.0.1</em>.
     *
     * @param address a valid hostname or textual representation of an IP address
     * @return same EmbeddedLdapRuleBuilder instance with the bindAddress field set
     * @throws IllegalArgumentException if the value provided for \"address\" is invalid
     */
    public EmbeddedLdapRuleBuilder bindingToAddress(final String address) {
        Objects.requireNonNull(address);
        try {
            final InetAddress addressByName = InetAddress.getByName(address);
            this.bindAddress = addressByName;
        } catch (UnknownHostException e) {
            throw new IllegalArgumentException(String.format("Unknown host address \"%s\"", address), e);
        }
        return this;
    }

    /**
     * Avoid adding UnboundID's default schema that contains the most common LDAP elements defined through various RFC's.
     *
     * @return same EmbeddedLdapRuleBuilder instance with the withoutDefaultSchema field set to FALSE
     */
    public EmbeddedLdapRuleBuilder withoutDefaultSchema() {
        this.addDefaultSchema = false;
        return this;
    }

    /**
     * Define schemas to be used for the server. If not defined, UnboundID will set up a default schema.
     *
     * @param ldifSchemaFiles LDIF-files containing schema element definitions
     * @return same EmbeddedLdapRuleBuilder with the given LDIF-files added to the internal schema file collection.
     */
    public EmbeddedLdapRuleBuilder withSchema(final String... ldifSchemaFiles) {
        this.schemaLdifs.addAll(Arrays.asList(ldifSchemaFiles));
        return this;
    }

    /**
     * Specify one or more LDIF resources to be imported on startup.
     *
     * @param ldifFiles LDIF-files to import
     * @return same EmbeddedLdapRuleBuilder instance with the provided ldifFiles added to the list of LDIF files to import
     */
    public EmbeddedLdapRuleBuilder importingLdifs(final String... ldifFiles) {
        if (ldifFiles != null) {
            ldifsToImport.addAll(Arrays.asList(ldifFiles));
        }
        return this;
    }

    /**
     * Creates a new rule based on the information that was previously provided
     *
     * @return a new EmbeddedLdapRule instance
     */
    public EmbeddedLdapRule build() {
        Objects.requireNonNull(bindDSN, "\"bindDSN\" can not be null");
        return EmbeddedLdapRuleImpl.createForConfiguration(createInMemoryServerConfiguration(),
                                                           authenticationConfiguration,
                                                           ldifsToImport);
    }

    private InMemoryDirectoryServerConfig createInMemoryServerConfiguration() {
        try {
            final InMemoryDirectoryServerConfig inMemoryDirectoryServerConfig =
                    new InMemoryDirectoryServerConfig(domainDsnArray());

            if (bindCredentials != null) {
                this.authenticationConfiguration = new AuthenticationConfiguration(bindDSN, bindCredentials);
                inMemoryDirectoryServerConfig.addAdditionalBindCredentials(bindDSN, bindCredentials);
            }

            final InMemoryListenerConfig listenerConfig = InMemoryListenerConfig.createLDAPConfig(
                    LDAP_SERVER_LISTENER_NAME,
                    bindAddress,
                    bindPort,
                    null);
            inMemoryDirectoryServerConfig.setListenerConfigs(listenerConfig);
            inMemoryDirectoryServerConfig.setSchema(customSchema());
            return inMemoryDirectoryServerConfig;
        } catch (LDAPException e) {
            throw new IllegalStateException(
                    "Could not create configuration for the in-memory LDAP instance due to an exception",
                    e);
        }
    }

    private String[] domainDsnArray() {
        if (domainDsn.size() == 0) {
            return new String[]{DEFAULT_DOMAIN};
        } else {
            return domainDsn.toArray(new String[]{});
        }
    }

    private Schema customSchema() {
        final List<File> schemaFiles = schemaFiles();

        try {
            final Schema initialSchema = (addDefaultSchema ? Schema.getDefaultStandardSchema() : null);
            if (!schemaFiles.isEmpty()) {
                final Schema customSchema = initialSchema == null
                                            ? Schema.getSchema(schemaFiles)
                                            : Schema.mergeSchemas(initialSchema, Schema.getSchema(schemaFiles));
                return customSchema;
            } else {
                return null;
            }

        } catch (IOException | LDIFException | LDAPException e) {
            throw new IllegalArgumentException(
                    "Could not create custom LDAP schema due, probably caused by an incorrectly formatted schema",
                    e);
        }
    }

    private List<File> schemaFiles() {
        return Lists.newArrayList(Lists.transform(this.schemaLdifs, new Function<String, File>() {
            @Override
            public File apply(final String input) {
                try {
                    final File file = new File(Resources.getResource(input).toURI());
                    if (!file.isFile()) {
                        throw new IllegalArgumentException(String.format(
                                "The resource named \"%s\" can not be found or is not a file",
                                input));
                    }
                    return file;
                } catch (URISyntaxException e) {
                    throw new IllegalArgumentException(String.format(
                            "The resource named \"%s\" is not a valid file reference",
                            input), e);
                }
            }
        }));
    }


}
