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
package org.apache.submarine.server.workbench.rest;

import com.google.gson.Gson;
import org.apache.hadoop.hive.metastore.api.Database;
import org.apache.hadoop.hive.metastore.api.InvalidInputException;
import org.apache.hadoop.hive.metastore.api.InvalidObjectException;
import org.apache.hadoop.hive.metastore.api.MetaException;
import org.apache.hadoop.hive.metastore.api.NoSuchObjectException;
import org.apache.hadoop.hive.metastore.api.Table;
import org.apache.submarine.commons.metastore.SubmarineMetaStore;
import org.apache.submarine.commons.utils.SubmarineConfiguration;
import org.apache.submarine.server.response.JsonResponse;
import org.apache.submarine.server.workbench.annotation.SubmarineApi;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.inject.Inject;
import javax.inject.Singleton;
import javax.ws.rs.DELETE;
import javax.ws.rs.GET;
import javax.ws.rs.POST;
import javax.ws.rs.PUT;
import javax.ws.rs.Path;
import javax.ws.rs.Produces;
import javax.ws.rs.QueryParam;
import javax.ws.rs.core.Response;
import java.util.List;

@Path("/metastore")
@Produces("application/json")
@Singleton
public class MetastoreRestApi {
  private static final Logger LOG = LoggerFactory.getLogger(MetastoreRestApi.class);

  private static final SubmarineConfiguration submarineConf = SubmarineConfiguration.newInstance();
  private SubmarineMetaStore submarineMetaStore = new SubmarineMetaStore(submarineConf);

  @Inject
  public MetastoreRestApi() {
  }

  @GET
  @Path("/databaseCount")
  @SubmarineApi
  public Response getDatabaseCount() {
    int databaseCount = 0;
    try {
      databaseCount = submarineMetaStore.getDatabaseCount();
    } catch (MetaException e) {
      LOG.error(e.getMessage(), e);
    }
    LOG.info("databaseCount:{}", databaseCount);
    return new JsonResponse.Builder<Integer>(Response.Status.OK)
        .success(true).result(databaseCount).build();
  }

  @GET
  @Path("/tableCount")
  @SubmarineApi
  public Response getTableCount() {
    int tableCount = 0;
    try {
      tableCount = submarineMetaStore.getTableCount();
    } catch (MetaException e) {
      LOG.error(e.getMessage(), e);
    }
    LOG.info("tableCount:{}", tableCount);
    return new JsonResponse.Builder<Integer>(Response.Status.OK)
        .success(true).result(tableCount).build();
  }

  @GET
  @Path("/allDatabases")
  @SubmarineApi
  public Response getAllDatabases() {
    List<String> databases = null;
    try {
      databases = submarineMetaStore.getAllDatabases();
    } catch (MetaException e) {
      LOG.error(e.getMessage(), e);
    }
    return new JsonResponse.Builder<List<String>>(Response.Status.OK)
        .success(true).result(databases).build();
  }

  @GET
  @Path("/database")
  @SubmarineApi
  public Response getDatabase(@QueryParam("databaseName") String databaseName) {
    Database database = null;
    try {
      database = submarineMetaStore.getDatabase(databaseName);
    } catch (NoSuchObjectException e) {
      LOG.error(e.getMessage(), e);
    }
    return new JsonResponse.Builder<Database>(Response.Status.OK)
        .success(true).result(database).build();
  }

  @GET
  @Path("/allTables")
  @SubmarineApi
  public Response getAllTables(@QueryParam("databaseName") String databaseName) {
    List<String> tables = null;
    try {
      tables = submarineMetaStore.getAllTables(databaseName);
    } catch (MetaException e) {
      LOG.error(e.getMessage(), e);
    }
    return new JsonResponse.Builder<List<String>>(Response.Status.OK)
        .success(true).result(tables).build();
  }

  @GET
  @Path("/table")
  @SubmarineApi
  public Response getTable(@QueryParam("databaseName") String databaseName,
                              @QueryParam("tableName") String tableName) {
    Table table = null;
    try {
      table = submarineMetaStore.getTable(databaseName, tableName);
    } catch (MetaException e) {
      LOG.error(e.getMessage(), e);
    }
    return new JsonResponse.Builder<Table>(Response.Status.OK)
        .success(true).result(table).build();
  }

  @POST
  @Path("/database")
  @SubmarineApi
  public Response createDatabase(Database database) {
    try {
      submarineMetaStore.createDatabase(database);
    } catch (MetaException | InvalidObjectException e) {
      LOG.error(e.getMessage(), e);
      return new JsonResponse.Builder<>(Response.Status.OK).success(false)
          .message("create database failed!").build();
    }
    return new JsonResponse.Builder<>(Response.Status.OK).success(true)
        .message("create database successfully!").build();
  }

  @PUT
  @Path("/database")
  @SubmarineApi
  public Response alterDatabase(@QueryParam("databaseName") String databaseName,
                                Database database) {
    boolean result = false;
    try {
      result = submarineMetaStore.alterDatabase(databaseName, database);
    } catch (MetaException | NoSuchObjectException e) {
      LOG.error(e.getMessage(), e);
      return new JsonResponse.Builder<>(Response.Status.OK).success(result)
          .message("alter database failed!").build();
    }
    return new JsonResponse.Builder<>(Response.Status.OK).success(result)
        .message("alter database successfully!").build();
  }

  @DELETE
  @Path("/database")
  @SubmarineApi
  public Response dropDatabase(@QueryParam("databaseName") String databaseName) {
    boolean result = false;
    try {
      result = submarineMetaStore.dropDatabase(databaseName);
    } catch (MetaException | NoSuchObjectException e) {
      LOG.error(e.getMessage(), e);
      return new JsonResponse.Builder<>(Response.Status.OK).success(result)
          .message("delete database failed!").build();
    }
    return new JsonResponse.Builder<>(Response.Status.OK).success(result)
        .message("delete database successfully!").build();
  }

  @POST
  @Path("/table")
  @SubmarineApi
  public Response createTable(String json) {
    Gson gson = new Gson();
    Table table = gson.fromJson(json, Table.class);
    try {
      submarineMetaStore.createTable(table);
    } catch (MetaException | InvalidObjectException e) {
      LOG.error(e.getMessage(), e);
      return new JsonResponse.Builder<>(Response.Status.OK).success(false)
          .message("create table failed!").build();
    }
    return new JsonResponse.Builder<>(Response.Status.OK).success(true)
        .message("create table successfully!").build();
  }

  @DELETE
  @Path("/table")
  @SubmarineApi
  public Response dropTable(@QueryParam("databaseName") String databaseName,
                            @QueryParam("tableName") String tableName) {
    boolean result = false;
    try {
      result = submarineMetaStore.dropTable(databaseName, tableName);
    } catch (MetaException | InvalidObjectException | NoSuchObjectException | InvalidInputException e) {
      LOG.error(e.getMessage(), e);
      return new JsonResponse.Builder<>(Response.Status.OK).success(result)
          .message("drop table failed!").build();
    }
    return new JsonResponse.Builder<>(Response.Status.OK).success(result)
        .message("drop table successfully!").build();
  }

}
