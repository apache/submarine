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
package org.apache.submarine.commons.metastore;

import org.apache.hadoop.hive.conf.HiveConf;
import org.apache.hadoop.hive.metastore.RawStore;
import org.apache.hadoop.hive.metastore.RawStoreProxy;
import org.apache.hadoop.hive.metastore.api.Database;
import org.apache.hadoop.hive.metastore.api.Index;
import org.apache.hadoop.hive.metastore.api.InvalidInputException;
import org.apache.hadoop.hive.metastore.api.InvalidObjectException;
import org.apache.hadoop.hive.metastore.api.MetaException;
import org.apache.hadoop.hive.metastore.api.NoSuchObjectException;
import org.apache.hadoop.hive.metastore.api.Partition;
import org.apache.hadoop.hive.metastore.api.SQLForeignKey;
import org.apache.hadoop.hive.metastore.api.SQLPrimaryKey;
import org.apache.hadoop.hive.metastore.api.Table;
import org.apache.hadoop.hive.metastore.api.TableMeta;
import org.apache.submarine.commons.utils.SubmarineConfiguration;
import parquet.org.slf4j.Logger;
import parquet.org.slf4j.LoggerFactory;

import java.util.List;

public class SubmarineMetaStore {
  private static final Logger LOG = LoggerFactory.getLogger(SubmarineMetaStore.class);

  private RawStore rs = null;

  public SubmarineMetaStore(SubmarineConfiguration submarineConf) {
    HiveConf conf = new HiveConf();
    conf.setVar(HiveConf.ConfVars.METASTORE_CONNECTION_DRIVER, submarineConf.getJdbcDriverClassName());
    conf.setVar(HiveConf.ConfVars.METASTORECONNECTURLKEY, submarineConf.getMetastoreJdbcUrl());
    conf.setVar(HiveConf.ConfVars.METASTORE_CONNECTION_USER_NAME, submarineConf.getMetastoreJdbcUserName());
    conf.setVar(HiveConf.ConfVars.METASTOREPWD, submarineConf.getMetastoreJdbcPassword());
    // Id can be set to any int value
    try {
      this.rs = RawStoreProxy.getProxy(conf, conf,
          conf.getVar(HiveConf.ConfVars.METASTORE_RAW_STORE_IMPL), 1);
    } catch (MetaException e) {
      LOG.error(e.getMessage(), e);
    }
  }

  /**
   * Gets total number of databases.
   * @return
   * @throws MetaException
   */
  public int getDatabaseCount() throws MetaException {
    int databaseCount = rs.getDatabaseCount();
    return databaseCount;
  }

  public List<String> getAllDatabases() throws MetaException {
    List<String> databases = rs.getAllDatabases();
    return databases;
  }

  public Database getDatabase(String databaseName) throws NoSuchObjectException {
    Database database = rs.getDatabase(databaseName);
    return database;
  }

  /**
   * Gets total number of tables.
   * @return
   * @throws MetaException
   */
  public int getTableCount() throws MetaException {
    int tableCount = rs.getTableCount();
    return tableCount;
  }

  public List<String> getAllTables(String databaseName) throws MetaException {
    List<String> tables = rs.getAllTables(databaseName);
    return tables;
  }

  public Table getTable(String databaseName, String tableName) throws MetaException {
    Table table = rs.getTable(databaseName, tableName);
    return table;
  }

  public void createDatabase(Database db) throws InvalidObjectException, MetaException {
    rs.createDatabase(db);
  }

  /**
   * Alter the database object in metastore. Currently only the parameters
   * of the database or the owner can be changed.
   * @param dbName the database name
   * @param db the Hive Database object
   * @throws MetaException
   * @throws NoSuchObjectException
   */
  public boolean alterDatabase(String dbName, Database db) throws NoSuchObjectException, MetaException {
    boolean result = rs.alterDatabase(dbName, db);
    return result;
  }

  public boolean dropDatabase(String dbName) throws NoSuchObjectException, MetaException {
    boolean result = rs.dropDatabase(dbName);
    return result;
  }

  public void createTable(Table table) throws InvalidObjectException, MetaException {
    rs.createTable(table);
  }

  public boolean dropTable(String dbName, String tableName)
      throws MetaException, InvalidObjectException, NoSuchObjectException, InvalidInputException {
    boolean result = rs.dropTable(dbName, tableName);
    return result;
  }

  public int getPartitionCount() throws MetaException {
    int partitionCount = rs.getPartitionCount();
    return partitionCount;
  }

  public Partition getPartition(String dbName, String tableName, List<String> partVals)
      throws NoSuchObjectException, MetaException {
    Partition partition = rs.getPartition(dbName, tableName, partVals);
    return partition;
  }

  public List<TableMeta> getTableMeta(String dbNames, String tableNames, List<String> tableTypes)
      throws MetaException {
    List<TableMeta> tableMetas = rs.getTableMeta(dbNames, tableNames, tableTypes);
    return tableMetas;
  }

  public void createTableWithConstraints(Table tbl,
                                         List<SQLPrimaryKey> primaryKeys,
                                         List<SQLForeignKey> foreignKeys)
      throws InvalidObjectException, MetaException {
    rs.createTableWithConstraints(tbl, primaryKeys, foreignKeys);
  }

  public boolean addPartitions(String dbName, String tblName, List<Partition> parts)
      throws InvalidObjectException, MetaException {
    boolean result = rs.addPartitions(dbName, tblName, parts);
    return result;
  }

  public boolean dropPartition(String dbName, String tableName, List<String> partVals)
      throws MetaException, NoSuchObjectException, InvalidObjectException,
      InvalidInputException {
    boolean result = rs.dropPartition(dbName, tableName, partVals);
    return result;
  }

  public void alterTable(String dbname, String tableName, Table newTable)
      throws InvalidObjectException, MetaException {
    rs.alterTable(dbname, tableName, newTable);
  }

  public void alterIndex(String dbname, String baseTblName, String indexName, Index newIndex)
      throws InvalidObjectException, MetaException {
    rs.alterIndex(dbname, baseTblName, indexName, newIndex);
  }

  public void alterPartition(String dbname, String tableName, List<String> partVals, Partition newPart)
      throws InvalidObjectException, MetaException {
    rs.alterPartition(dbname, tableName, partVals, newPart);
  }

  public void addPrimaryKeys(List<SQLPrimaryKey> pks) throws InvalidObjectException, MetaException {
    rs.addPrimaryKeys(pks);
  }

  public boolean addIndex(Index index) throws InvalidObjectException, MetaException {
    boolean result = rs.addIndex(index);
    return result;
  }

  public boolean dropIndex(String dbName, String origTableName, String indexName) throws MetaException {
    boolean result = rs.dropIndex(dbName, origTableName, indexName);
    return result;
  }

  public Index getIndex(String dbName, String origTableName, String indexName) throws MetaException {
    Index index = rs.getIndex(dbName, origTableName, indexName);
    return index;
  }
}
