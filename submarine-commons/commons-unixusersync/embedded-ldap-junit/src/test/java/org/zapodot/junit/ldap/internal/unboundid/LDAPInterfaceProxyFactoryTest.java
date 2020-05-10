package org.zapodot.junit.ldap.internal.unboundid;

import org.junit.Test;

import java.lang.reflect.Constructor;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;

public class LDAPInterfaceProxyFactoryTest {

    @Test(expected = IllegalAccessException.class)
    public void constructorIsPrivate() throws Exception {
        final Constructor<LDAPInterfaceProxyFactory> declaredConstructor = LDAPInterfaceProxyFactory.class
                .getDeclaredConstructor();
        assertNotNull(declaredConstructor);
        assertFalse(declaredConstructor.isAccessible());
        declaredConstructor.newInstance();
    }

    @Test
    public void invokeInstructorAnyway() throws Exception {

        // Added to increase test coverage after adding a private constructor
        final Constructor<LDAPInterfaceProxyFactory> declaredConstructor = LDAPInterfaceProxyFactory.class
                .getDeclaredConstructor();
        assertNotNull(declaredConstructor);
        assertFalse(declaredConstructor.isAccessible());
        declaredConstructor.setAccessible(true);
        try {
            assertNotNull(declaredConstructor.newInstance());
        } finally {
            declaredConstructor.setAccessible(false);
        }
    }
}