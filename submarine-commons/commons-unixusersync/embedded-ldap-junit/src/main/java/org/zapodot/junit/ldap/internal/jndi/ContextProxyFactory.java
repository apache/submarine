package org.zapodot.junit.ldap.internal.jndi;

import net.bytebuddy.ByteBuddy;
import net.bytebuddy.NamingStrategy;
import net.bytebuddy.description.method.MethodDescription;
import net.bytebuddy.description.modifier.Visibility;
import net.bytebuddy.description.type.TypeDescription;
import net.bytebuddy.dynamic.loading.ClassLoadingStrategy;
import net.bytebuddy.implementation.FieldAccessor;
import net.bytebuddy.implementation.MethodDelegation;
import net.bytebuddy.matcher.ElementMatchers;

import javax.naming.Context;
import javax.naming.directory.DirContext;
import javax.naming.directory.InitialDirContext;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;

import static net.bytebuddy.matcher.ElementMatchers.isDeclaredBy;
import static net.bytebuddy.matcher.ElementMatchers.named;
import static net.bytebuddy.matcher.ElementMatchers.not;

/**
 * A factory that creates delegating proxys for the Context and DirContext interfaces that delegates to an underlying InitialDirContext
 * <p>
 * This class is part of the internal API and may thus be changed or removed without warning
 */
public class ContextProxyFactory {

    private static final String DELEGATED_CONTEXT_FIELD_NAME = "delegatedContext";

    private static final String DELEGATED_DIR_CONTEXT_FIELD_NAME = "delegatedDirContext";

    private static final String DELEGATING_DIR_CONTEXT_PREFIX = "DelegatingDirContext";

    private ContextProxyFactory() {
    }

    private static final Class<? extends Context> CONTEXT_PROXY_TYPE =
            new ByteBuddy().subclass(Context.class)
                           .name(new NamingStrategy.PrefixingRandom("DelegatingContext")
                                         .subclass(new TypeDescription.Generic.OfNonGenericType.ForLoadedType(Context.class)))
                           .method(ElementMatchers.<MethodDescription>isDeclaredBy(Context.class)
                                           .and(not(ElementMatchers.<MethodDescription>named("close")))
                                           .and(not(ElementMatchers.<MethodDescription>isNative())))
                           .intercept(MethodDelegation.toField(DELEGATED_CONTEXT_FIELD_NAME))
                           .defineField(DELEGATED_CONTEXT_FIELD_NAME, Context.class, Visibility.PRIVATE)
                           .method(isDeclaredBy(Context.class).and(named("close")))
                           .intercept(MethodDelegation.to(ContextInterceptor.class))
                           .implement(ContextProxy.class)
                           .intercept(FieldAccessor.ofBeanProperty())
                           .make()
                           .load(ContextProxyFactory.class
                                         .getClassLoader(),
                                 ClassLoadingStrategy.Default.WRAPPER)
                           .getLoaded();

    private static final Class<? extends DirContext> DIR_CONTEXT_PROXY_TYPE =
            new ByteBuddy().subclass(DirContext.class)
                           .name(new NamingStrategy.PrefixingRandom(
                                   DELEGATING_DIR_CONTEXT_PREFIX)
                                         .subclass(new TypeDescription.Generic.OfNonGenericType.ForLoadedType(DirContext.class)))
                           .method(isDeclaredBy(
                                   DirContext.class))
                           .intercept(MethodDelegation
                                              .toField(DELEGATED_DIR_CONTEXT_FIELD_NAME))
                           .defineField(DELEGATED_DIR_CONTEXT_FIELD_NAME, DirContext.class, Visibility.PRIVATE)
                           .implement(DirContextProxy.class)
                           .intercept(FieldAccessor.ofBeanProperty())
                           .make()
                           .load(ContextProxyFactory.class
                                         .getClassLoader(),
                                 ClassLoadingStrategy.Default.WRAPPER)
                           .getLoaded();

    public static Context asDelegatingContext(final InitialDirContext initialDirContext) {
        return createProxy(initialDirContext);
    }

    private static Context createProxy(final InitialDirContext initialDirContext) {

        try {
            final Context contextDelegator = getDeclaredConstructor().newInstance();
            ((ContextProxy) contextDelegator).setDelegatedContext(initialDirContext);
            return contextDelegator;
        } catch (InstantiationException | IllegalAccessException | InvocationTargetException e) {
            throw new IllegalStateException(e);
        }
    }

    private static Constructor<? extends Context> getDeclaredConstructor() {
        try {
            return CONTEXT_PROXY_TYPE.getDeclaredConstructor();
        } catch (NoSuchMethodException e) {
            throw new IllegalStateException("Can not find a default constructor for proxy class", e);
        }
    }

    public static DirContext asDelegatingDirContext(final InitialDirContext initialDirContext) {
        try {
            final DirContext dirContext = DIR_CONTEXT_PROXY_TYPE.newInstance();
            ((DirContextProxy) dirContext).setDelegatedDirContext(initialDirContext);
            return dirContext;
        } catch (InstantiationException | IllegalAccessException e) {
            throw new IllegalStateException("Could not wrap DirContext", e);
        }
    }

}
