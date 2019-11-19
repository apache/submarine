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

import org.apache.hadoop.hive.conf.HiveConf;
import org.apache.hadoop.hive.metastore.api.Database;
import org.apache.hadoop.hive.metastore.api.FieldSchema;
import org.apache.hadoop.hive.metastore.api.InvalidObjectException;
import org.apache.hadoop.hive.metastore.api.MetaException;
import org.apache.hadoop.hive.metastore.api.NoSuchObjectException;
import org.apache.hadoop.hive.metastore.api.PrincipalType;
import org.apache.hadoop.hive.metastore.api.SerDeInfo;
import org.apache.hadoop.hive.metastore.api.StorageDescriptor;
import org.apache.hadoop.hive.metastore.api.Table;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;

public class SubmarineMetaStoreTest {
  private static HiveConf conf = new HiveConf();

  static {
    conf.setVar(HiveConf.ConfVars.METASTORE_CONNECTION_DRIVER, "com.mysql.jdbc.Driver");
    conf.setVar(HiveConf.ConfVars.METASTORECONNECTURLKEY, "jdbc:mysql://127.0.0.1:3306/metastoreDB_test?" +
        "useUnicode=true&amp;characterEncoding=UTF-8&amp;autoReconnect=true&amp;" +
        "failOverReadOnly=false&amp;zeroDateTimeBehavior=convertToNull&amp;useSSL=false");
    conf.setVar(HiveConf.ConfVars.METASTORE_CONNECTION_USER_NAME, "metastore_test");
    conf.setVar(HiveConf.ConfVars.METASTOREPWD, "password_test");
  }

  private SubmarineMetaStore submarineMetaStore = new SubmarineMetaStore(conf);

  @Before
  public void createDatabase() throws InvalidObjectException, MetaException {
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
  }

  @After
  public void removeAllRecord() throws Exception {
    submarineMetaStore.dropTable("testdb", "testtable");
    int tableCount = submarineMetaStore.getTableCount();
    assertEquals(0, tableCount);

    submarineMetaStore.dropDatabase("testdb");
    assertEquals(1, submarineMetaStore.getDatabaseCount());
  }

  @Test
  public void getDatabaseCount() throws InvalidObjectException, MetaException {
    assertEquals(2, submarineMetaStore.getDatabaseCount());
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
