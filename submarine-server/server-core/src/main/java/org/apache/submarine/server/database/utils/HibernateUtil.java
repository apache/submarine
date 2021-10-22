package org.apache.submarine.server.database.utils;


import org.apache.submarine.commons.runtime.exception.SubmarineRuntimeException;
import org.hibernate.SessionFactory;
import org.hibernate.boot.MetadataSources;
import org.hibernate.boot.registry.StandardServiceRegistry;
import org.hibernate.boot.registry.StandardServiceRegistryBuilder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class HibernateUtil {
  private static final Logger LOG = LoggerFactory.getLogger(HibernateUtil.class);

  private static final SessionFactory sessionFactory = buildSessionFactory();

  public static void close() {
    System.out.println("session " + sessionFactory);
    if (sessionFactory != null){
      sessionFactory.close();
    }
  }

  public static SessionFactory getSessionFactory() {
    return sessionFactory;
  }

  private static SessionFactory buildSessionFactory() throws SubmarineRuntimeException {
    final StandardServiceRegistry registry = new StandardServiceRegistryBuilder().configure().build();
    try {
      return new MetadataSources(registry).buildMetadata().buildSessionFactory();
    } catch (Exception e) {
      StandardServiceRegistryBuilder.destroy(registry);
      LOG.error(e.getMessage(), e);
      throw new SubmarineRuntimeException("Unable to build session factory");
    }
  }

  private HibernateUtil() {}
}
