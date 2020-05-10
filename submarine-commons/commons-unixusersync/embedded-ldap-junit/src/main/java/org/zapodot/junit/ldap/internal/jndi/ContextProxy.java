package org.zapodot.junit.ldap.internal.jndi;

import javax.naming.Context;

/**
 * Interface that is dynamically added to the Context proxy class to allow delegation.
 *
 * This interface is part of the internal API and may thus be changed or removed without warning.
 */
public interface ContextProxy {

    void setDelegatedContext(final Context context);
}
