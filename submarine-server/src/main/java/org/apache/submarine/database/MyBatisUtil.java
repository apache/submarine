/**
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. See accompanying LICENSE file.
 */
package org.apache.submarine.database;

import org.apache.ibatis.io.Resources;
import org.apache.ibatis.session.SqlSessionFactory;
import org.apache.ibatis.session.SqlSessionFactoryBuilder;
import org.apache.submarine.server.SubmarineConfiguration;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.Reader;
import java.util.Properties;

public class MyBatisUtil {
  private static final Logger LOG = LoggerFactory.getLogger(MyBatisUtil.class);

  private static SqlSessionFactory factory = null;

  public static SqlSessionFactory getSqlSessionFactory() {
    synchronized (MyBatisUtil.class) {
      if (null == factory) {
        Reader reader = null;
        try {
          reader = Resources.getResourceAsReader("mybatis/mybatis-config.xml");
        } catch (IOException e) {
          LOG.error(e.getMessage(), e);
          throw new RuntimeException(e.getMessage());
        }

        SubmarineConfiguration conf = SubmarineConfiguration.create();
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

        factory = new SqlSessionFactoryBuilder().build(reader, props);
      }
      return factory;
    }
  }
}
