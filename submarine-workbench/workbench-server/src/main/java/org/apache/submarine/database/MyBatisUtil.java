/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
package org.apache.submarine.database;

import org.apache.ibatis.io.Resources;
import org.apache.ibatis.session.SqlSession;
import org.apache.ibatis.session.SqlSessionFactory;
import org.apache.ibatis.session.SqlSessionFactoryBuilder;
import org.apache.submarine.commons.utils.SubmarineConfiguration;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.Reader;
import java.util.Properties;

public class MyBatisUtil {
  private static final Logger LOG = LoggerFactory.getLogger(MyBatisUtil.class);

  private static SqlSessionFactory sqlSessionFactory;

  static {
    Reader reader = null;
    try {
      try {
        reader = Resources.getResourceAsReader("mybatis-config.xml");
      } catch (IOException e) {
        LOG.error(e.getMessage(), e);
        throw new RuntimeException(e.getMessage());
      }

      checkCalledByTestMethod();

      SubmarineConfiguration conf = SubmarineConfiguration.getInstance();
      String jdbcClassName = conf.getJdbcDriverClassName();
      String jdbcUrl = conf.getJdbcUrl();
      String jdbcUserName = conf.getJdbcUserName();
      String jdbcPassword = conf.getJdbcPassword();
      LOG.info("MyBatisUtil -> jdbcClassName: {}, jdbcUrl: {}, jdbcUserName: {}, jdbcPassword: {}",
          jdbcClassName, jdbcUrl, jdbcUserName, jdbcPassword);

      Properties props = new Properties();
      props.setProperty("jdbc.driverClassName", jdbcClassName);
      props.setProperty("jdbc.url", jdbcUrl);
      props.setProperty("jdbc.username", jdbcUserName);
      props.setProperty("jdbc.password", jdbcPassword);

      sqlSessionFactory = new SqlSessionFactoryBuilder().build(reader, props);
    } finally {
      try {
        if (null != reader) {
          reader.close();
        }
      } catch (IOException e) {
        LOG.error(e.getMessage(), e);
      }
    }
  }

  /**
   * Get Session
   *
   * @return
   */
  public static SqlSession getSqlSession() {
    return sqlSessionFactory.openSession();
  }

  private static void checkCalledByTestMethod() {
    StackTraceElement[] stackTraceElements = Thread.currentThread().getStackTrace();
    for (StackTraceElement element : stackTraceElements) {
      if (element.getClassName().endsWith("Test")) {
        usingTestDatabase();
        return;
      }
    }
  }

  private static void usingTestDatabase() {
    LOG.info("Run the test unit using the test database");
    // Run the test unit using the test database
    SubmarineConfiguration conf = SubmarineConfiguration.getInstance();
    conf.setJdbcUrl("jdbc:mysql://127.0.0.1:3306/submarineDB_test?" +
        "useUnicode=true&amp;characterEncoding=UTF-8&amp;autoReconnect=true&amp;" +
        "failOverReadOnly=false&amp;zeroDateTimeBehavior=convertToNull&amp;useSSL=false");
    conf.setJdbcUserName("submarine_test");
    conf.setJdbcPassword("password_test");
  }
}
