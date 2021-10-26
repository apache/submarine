/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
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
    if (sessionFactory != null){
      sessionFactory.close();
    }
    LOG.info("Hibernate session is closed.");
  }

  public static SessionFactory getSessionFactory() {
    return sessionFactory;
  }

  private static SessionFactory buildSessionFactory() throws SubmarineRuntimeException {
    // Default get the hibernate.cfg.xml in resource
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
