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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.io.IOException;
import java.io.Reader;
import java.util.Properties;

public class ModelBatisUtil {
  private static final Logger LOG = LoggerFactory.getLogger(ModelBatisUtil.class);

  private static final SqlSessionFactory sqlSessionFactory;

  static {

    try (Reader reader =
                 Resources.getResourceAsReader("modelbatis-config.xml");
    ) {
      String jdbcClassName = "com.mysql.jdbc.Driver";
      String jdbcUrl = "jdbc:mysql://127.0.0.1:3306/mlflowdb";
      String jdbcUserName = "mlflow";
      String jdbcPassword = "password";
      LOG.info("MyBatisUtil -> jdbcClassName: {}, jdbcUrl: {}, jdbcUserName: {}, jdbcPassword: {}",
          jdbcClassName, jdbcUrl, jdbcUserName, jdbcPassword);

      Properties props = new Properties();
      props.setProperty("jdbc.driverClassName", jdbcClassName);
      props.setProperty("jdbc.url", jdbcUrl);
      props.setProperty("jdbc.username", jdbcUserName);
      props.setProperty("jdbc.password", jdbcPassword);

      sqlSessionFactory = new SqlSessionFactoryBuilder().build(reader, props);
    } catch (IOException e) {
      LOG.error(e.getMessage(), e);
      throw new RuntimeException(e.getMessage());
    }
  }

  /**
   * Get Session.
   *
   * @return SqlSession
   */
  public static SqlSession getSqlSession() {
    return sqlSessionFactory.openSession();
  }


}
