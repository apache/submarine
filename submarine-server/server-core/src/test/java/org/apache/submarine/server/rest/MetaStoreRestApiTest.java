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
package org.apache.submarine.server.rest;

import com.google.gson.Gson;
import org.apache.hadoop.hive.metastore.api.Database;
import org.apache.hadoop.hive.metastore.api.PrincipalType;
import org.apache.hadoop.hive.metastore.api.StorageDescriptor;
import org.apache.hadoop.hive.metastore.api.Table;
import org.apache.hadoop.hive.metastore.api.FieldSchema;
import org.apache.hadoop.hive.metastore.api.SerDeInfo;
import org.apache.submarine.commons.utils.SubmarineConfiguration;
import org.junit.After;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

import javax.ws.rs.core.Response;
import java.util.Date;
import java.util.Map;
import java.util.HashMap;
import java.util.List;
import java.util.ArrayList;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class MetaStoreRestApiTest {
  private static MetaStoreRestApi metaStoreApi;

  @BeforeClass
  public static void init() {
    SubmarineConfiguration submarineConf = SubmarineConfiguration.getInstance();
    submarineConf.setMetastoreJdbcUrl("jdbc:mysql://127.0.0.1:3306/metastore_test?" +
                                              "useUnicode=true&amp;" +
                                              "characterEncoding=UTF-8&amp;" +
                                              "autoReconnect=true&amp;" +
                                              "failOverReadOnly=false&amp;" +
                                              "zeroDateTimeBehavior=convertToNull&amp;" +
                                              "useSSL=false");
    submarineConf.setMetastoreJdbcUserName("metastore_test");
    submarineConf.setMetastoreJdbcPassword("password_test");
    metaStoreApi = new MetaStoreRestApi();
  }

  @Before
  public void createDatabase() {
    Database database = new Database();
    database.setName("testdb");
    database.setDescription("testdb");
    database.setLocationUri("hdfs://mycluster/user/hive/warehouse/testdb.db");
    Map<String, String> map = new HashMap<>();
    map.put("key", "value");
    database.setParameters(map);
    database.setOwnerName("root");
    database.setOwnerType(PrincipalType.USER);

    Gson gson = new Gson();
    String databaseJson = gson.toJson(database);

    metaStoreApi.createDatabase(databaseJson);
    Response databaseCountResponse = metaStoreApi.getDatabaseCount();
    assertEquals(databaseCountResponse.getStatus(), Response.Status.OK.getStatusCode());
    assertTrue(((String) databaseCountResponse.getEntity()).contains("\"result\":1"));

    Table table = new Table();
    table.setTableName("testtable");
    table.setDbName("testdb");
    table.setOwner("root");
    table.setCreateTime((int) new java.util.Date().getTime() / 1000);
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
    Map<String, String> parametersMap = new HashMap<>();
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

    String tableJson = gson.toJson(table);
    metaStoreApi.createTable(tableJson);

    Response tableResponse = metaStoreApi.getTable("testdb", "testtable");
    assertEquals(tableResponse.getStatus(), Response.Status.OK.getStatusCode());
    assertTrue(((String) tableResponse.getEntity()).contains("\"tableName\":\"testtable\""));
    Response tableCountResponse = metaStoreApi.getTableCount();
    assertEquals(tableCountResponse.getStatus(), Response.Status.OK.getStatusCode());
    assertTrue(((String) tableCountResponse.getEntity()).contains("\"result\":1"));
  }

  @After
  public void removeAllRecord() {
    metaStoreApi.dropTable("testdb", "testtable");
    Response tableCountResponse = metaStoreApi.getTableCount();
    assertEquals(tableCountResponse.getStatus(), Response.Status.OK.getStatusCode());
    assertTrue(((String) tableCountResponse.getEntity()).contains("\"result\":0"));

    metaStoreApi.dropDatabase("testdb");
    Response databaseCountResponse = metaStoreApi.getDatabaseCount();
    assertEquals(databaseCountResponse.getStatus(), Response.Status.OK.getStatusCode());
    assertTrue(((String) databaseCountResponse.getEntity()).contains("\"result\":0"));
  }

  @Test
  public void getDatabaseCount() {
    Response response = metaStoreApi.getDatabaseCount();
    assertTrue(((String) response.getEntity()).contains("\"result\":1"));
  }

  @Test
  public void listDatabases() {
    Response response = metaStoreApi.listDatabases();
    assertEquals(response.getStatus(), Response.Status.OK.getStatusCode());
    assertTrue(((String) response.getEntity()).contains("testdb"));
  }

  @Test
  public void getDatabase() {
    Response response = metaStoreApi.getDatabase("testdb");
    assertEquals(response.getStatus(), Response.Status.OK.getStatusCode());
    assertTrue(((String) response.getEntity()).contains("testdb"));
  }

  @Test
  public void getAllTables() {
    Response response = metaStoreApi.listTables("testdb");
    assertEquals(response.getStatus(), Response.Status.OK.getStatusCode());
    assertTrue(((String) response.getEntity()).contains("testtable"));
  }

  @Test
  public void getTableCount() {
    Response response = metaStoreApi.getTableCount();
    assertEquals(response.getStatus(), Response.Status.OK.getStatusCode());
    assertTrue(((String) response.getEntity()).contains("\"result\":1"));
  }

}
