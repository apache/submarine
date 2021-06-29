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

package org.apache.submarine.commons.metastore;

import org.apache.hadoop.hive.metastore.api.Database;
import org.apache.hadoop.hive.metastore.api.FieldSchema;
import org.apache.hadoop.hive.metastore.api.InvalidObjectException;
import org.apache.hadoop.hive.metastore.api.MetaException;
import org.apache.hadoop.hive.metastore.api.NoSuchObjectException;
import org.apache.hadoop.hive.metastore.api.PrincipalType;
import org.apache.hadoop.hive.metastore.api.SerDeInfo;
import org.apache.hadoop.hive.metastore.api.StorageDescriptor;
import org.apache.hadoop.hive.metastore.api.Table;
import org.apache.submarine.commons.utils.SubmarineConfiguration;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;

public class SubmarineMetaStoreTest {
  private static final Logger LOG = LoggerFactory.getLogger(SubmarineMetaStoreTest.class);
  private static final SubmarineConfiguration submarineConf = SubmarineConfiguration.newInstance();
  private SubmarineMetaStore submarineMetaStore = new SubmarineMetaStore(submarineConf);

  static {
    submarineConf.setMetastoreJdbcUrl("jdbc:mysql://127.0.0.1:3306/metastore_test?" +
        "useUnicode=true&characterEncoding=UTF-8&autoReconnect=true&" +
        "failOverReadOnly=false&zeroDateTimeBehavior=convertToNull&useSSL=false");
    submarineConf.setMetastoreJdbcUserName("metastore_test");
    submarineConf.setMetastoreJdbcPassword("password_test");
  }

  @Test
  public void listTables() {
    LOG.info("listTables >>> ");

    String url = "jdbc:mysql://127.0.0.1:3306/metastore_test?" +
        "useUnicode=true&characterEncoding=UTF-8&autoReconnect=true&" +
        "failOverReadOnly=false&zeroDateTimeBehavior=convertToNull&useSSL=false";
    String username = "metastore_test";
    String password = "password_test";
    boolean flag = false;
    Connection con = null;
    Statement stmt = null;
    try {
      con = DriverManager.getConnection(url, username, password);
      stmt = con.createStatement();
      String sql = "show tables";
      LOG.info(">>>>> sql:" + sql);
      ResultSet rs = stmt.executeQuery(sql);
      LOG.info("rs:" + rs);

      while (rs.next()) {
        String pass = rs.getString(1);
        LOG.info("table:" + pass);
      }
    } catch (SQLException se) {
      LOG.error(se.getMessage(), se);
    }

    LOG.info("listTables <<< ");
  }

  @Before
  public void createDatabase() throws InvalidObjectException, MetaException {
    listTables();

    Database database = new Database();
    database.setName("testdb");
    database.setDescription("testdb");
    database.setLocationUri("hdfs://mycluster/user/hive/warehouse/testdb.db");
    Map map = new HashMap();
    map.put("key", "value");
    database.setParameters(map);
    database.setOwnerName("root");
    database.setOwnerType(PrincipalType.USER);
    submarineMetaStore.createDatabase(database);
    assertEquals(1, submarineMetaStore.getDatabaseCount());

    Table table = new Table();
    table.setTableName("testtable");
    table.setDbName("testdb");
    table.setOwner("root");
    table.setCreateTime((int) new Date().getTime() / 1000);
    table.setLastAccessTime((int) new Date().getTime() / 1000);
    table.setRetention(0);
    StorageDescriptor sd = new StorageDescriptor();
    List<FieldSchema> fieldSchemas = new ArrayList<>();
    FieldSchema fieldSchema = new FieldSchema();
    fieldSchema.setName("a");
    fieldSchema.setType("int");
    fieldSchema.setComment("a");
    fieldSchemas.add(fieldSchema);
    sd.setCols(fieldSchemas);
    sd.setLocation("hdfs://mycluster/user/hive/warehouse/testdb.db/testtable");
    sd.setInputFormat("org.apache.hadoop.mapred.TextInputFormat");
    sd.setOutputFormat("org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat");
    sd.setCompressed(false);
    sd.setNumBuckets(-1);
    SerDeInfo serdeInfo = new SerDeInfo();
    serdeInfo.setName("test");
    serdeInfo.setSerializationLib("org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe");
    Map<String, String> parametersMap = new HashMap();
    parametersMap.put("serialization.format", "|");
    parametersMap.put("field.delim", "|");
    serdeInfo.setParameters(parametersMap);
    sd.setSerdeInfo(serdeInfo);
    table.setSd(sd);
    List<FieldSchema> partitionKeys = new ArrayList<>();
    table.setPartitionKeys(partitionKeys);
    Map<String, String> parameters = new HashMap<>();
    table.setParameters(parameters);
    String viewOriginalText = "";
    table.setViewOriginalText(viewOriginalText);
    String viewExpandedText = "";
    table.setViewExpandedText(viewExpandedText);
    String tableType = "MANAGED_TABLE";
    table.setTableType(tableType);
    submarineMetaStore.createTable(table);

    Table tableTest = submarineMetaStore.getTable("testdb", "testtable");
    assertEquals("testtable", tableTest.getTableName());
    int tableCount = submarineMetaStore.getTableCount();
    assertEquals(1, tableCount);
  }

  @After
  public void removeAllRecord() throws Exception {
    submarineMetaStore.dropTable("testdb", "testtable");
    int tableCount = submarineMetaStore.getTableCount();
    assertEquals(0, tableCount);

    submarineMetaStore.dropDatabase("testdb");
    assertEquals(0, submarineMetaStore.getDatabaseCount());
  }

  @Test
  public void getDatabaseCount() throws InvalidObjectException, MetaException {
    assertEquals(1, submarineMetaStore.getDatabaseCount());
  }

  @Test
  public void getAllDatabases() throws MetaException {
    List<String> databases = submarineMetaStore.getAllDatabases();
    assertEquals(true, databases.contains("testdb"));
  }

  @Test
  public void getDatabase() throws NoSuchObjectException {
    Database database = submarineMetaStore.getDatabase("testdb");
    assertEquals("testdb", database.getName());
  }

  @Test
  public void getAllTables() throws MetaException {
    List<String> tables = submarineMetaStore.getAllTables("testdb");
    assertEquals(true, tables.contains("testtable"));
  }

  @Test
  public void getTableCount() throws MetaException {
    int tableCount = submarineMetaStore.getTableCount();
    assertEquals(1, tableCount);
  }

}
