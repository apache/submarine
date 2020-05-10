package org.zapodot.junit.ldap.internal.jndi;

import javax.naming.directory.DirContext;

/**
 * An interface that is implemented by the DirContext proxy to allow delegation.
 *
 * This class is part of the internal API and may thus be changed or removed without warning.
 */
public interface DirContextProxy {

    void setDelegatedDirContext(final DirContext dirContext);
}
