package org.zapodot.junit.ldap.internal.jndi;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.naming.NamingException;

/**
 * Interceptor for the Context interface proxy that suppress calls to the close() method.
 * All other method calls is delegated to a "real" Context.
 *
 * This class is part of the internal API and may thus be changed or removed without warning.
 */
public class ContextInterceptor {

    private static final Logger LOGGER = LoggerFactory.getLogger(ContextInterceptor.class);

    public static void close() throws NamingException {
        LOGGER.debug("close() call intercepted. Will be ignored");
    }
}
