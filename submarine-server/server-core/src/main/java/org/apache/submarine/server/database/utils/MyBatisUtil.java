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
package org.apache.submarine.server.database.utils;

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
  private static SqlSessionFactory metastoreSqlSessionFactory;

  static {
    SubmarineConfiguration conf = SubmarineConfiguration.getInstance();
    sqlSessionFactory = buildSqlSessionFactory("mybatis-config.xml",
        conf.getJdbcUrl(), conf.getJdbcUserName(), conf.getJdbcPassword());
  }

  private static SqlSessionFactory buildSqlSessionFactory(String configFile,
      String jdbcUrl, String jdbcUserName, String jdbcPassword) {
    Reader reader = null;
    SqlSessionFactory sqlSessionFactory = null;
    try {
      try {
        reader = Resources.getResourceAsReader(configFile);
      } catch (IOException e) {
        LOG.error(e.getMessage(), e);
        throw new RuntimeException(e.getMessage());
      }
      checkCalledByTestMethod(jdbcUrl, jdbcUserName, jdbcPassword);
      String jdbcClassName =
          SubmarineConfiguration.getInstance().getJdbcDriverClassName();
      LOG.info(
          "MyBatisUtil -> jdbcClassName: {}, jdbcUrl: {}, jdbcUserName: {}, jdbcPassword: {}",
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
    return sqlSessionFactory;
  }

  /**
   * Get Session
   *
   * @return
   */
  public static SqlSession getSqlSession() {
    return sqlSessionFactory.openSession();
  }
  
  public static SqlSession getMetastoreSqlSession() {
    return metastoreSqlSessionFactory.openSession();
  }

  private static void checkCalledByTestMethod(String jdbcUrl,
      String jdbcUserName, String jdbcPassword) {
    StackTraceElement[] stackTraceElements =
        Thread.currentThread().getStackTrace();
    for (StackTraceElement element : stackTraceElements) {
      if (element.getClassName().endsWith("Test")) {
        usingTestDatabase(jdbcUrl, jdbcUserName, jdbcPassword);
        return;
      }
    }
  }

  private static void usingTestDatabase(String jdbcUrl, String jdbcUserName,
      String jdbcPassword) {
    LOG.info("Run the test unit using the test database");
    String jdbcPropertiesSuffix = "_test";
    String finalJdbcUrl = jdbcUrl.replace("?", jdbcPropertiesSuffix + "?");
    SubmarineConfiguration conf = SubmarineConfiguration.getInstance();
    conf.setJdbcUrl(finalJdbcUrl);
    conf.setJdbcUserName(jdbcUserName + jdbcPropertiesSuffix);
    conf.setJdbcPassword(jdbcPassword + jdbcPropertiesSuffix);
  }
}
