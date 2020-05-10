package org.zapodot.junit.ldap.internal.jndi;

import org.junit.Test;

import java.lang.reflect.Constructor;

import static org.junit.Assert.assertNotNull;

public class ContextProxyFactoryTest {

    @Test(expected = IllegalAccessException.class)
    public void couldNotInstantiate() throws Exception {
        final Constructor<ContextProxyFactory> declaredConstructor = ContextProxyFactory.class.getDeclaredConstructor();
        assertNotNull(declaredConstructor);
        declaredConstructor.newInstance();
    }

    @Test
    public void instantiateAnyWay() throws Exception {
        // This is only added to improve test coverage
        final Constructor<ContextProxyFactory> declaredConstructor = ContextProxyFactory.class.getDeclaredConstructor();
        assertNotNull(declaredConstructor);
        declaredConstructor.setAccessible(true);
        assertNotNull(declaredConstructor.newInstance());

    }
}