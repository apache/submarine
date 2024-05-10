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

package org.apache.submarine.server.database;

import org.apache.submarine.commons.utils.SubmarineConfiguration;
import org.apache.submarine.server.database.initialization.DatabaseMetadataService;
import org.junit.Before;
import org.junit.Test;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;

import static org.junit.Assert.assertTrue;

public class TestInitDatabaseMetadata {

  @Before
  public void setDefaultVars() {
    SubmarineConfiguration conf = SubmarineConfiguration.getInstance();
    conf.updateConfiguration("submarine.metadata.init", "true");
    conf.updateConfiguration("submarine.metadata.location", "classpath:test_db/migration");
    conf.updateConfiguration("jdbc.driverClassName", "org.h2.Driver");
    conf.updateConfiguration("jdbc.url", "jdbc:h2:mem:testdb;MODE=MYSQL;DB_CLOSE_DELAY=-1");
    conf.updateConfiguration("jdbc.username", "root");
    conf.updateConfiguration("jdbc.password", "");
  }

  @Test
  public void testInit() throws SQLException {
    // init database
    new DatabaseMetadataService().initDatabaseMetadata();
    // test table exists
    try (Connection conn = DriverManager.getConnection("jdbc:h2:mem:testdb;MODE=MYSQL;DB_CLOSE_DELAY=-1",
        "root", "");
         Statement stmt = conn.createStatement();
         ResultSet rs = stmt.executeQuery("select * from sys_user")) {
      assertTrue(rs.next());
    }
  }
}
