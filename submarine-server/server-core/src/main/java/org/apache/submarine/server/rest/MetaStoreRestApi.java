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
import com.google.gson.JsonSyntaxException;
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
import javax.ws.rs.Path;
import javax.ws.rs.Produces;
import javax.ws.rs.QueryParam;
import javax.ws.rs.core.Response;
import java.util.List;

@Path(RestConstants.V1 + "/" + RestConstants.METASTORE)
@Produces("application/json")
@Singleton
public class MetaStoreRestApi {
  private static final Logger LOG = LoggerFactory.getLogger(MetaStoreRestApi.class);
  private static final Gson gson = new Gson();
  private static final SubmarineConfiguration submarineConf = SubmarineConfiguration.getInstance();
  private SubmarineMetaStore submarineMetaStore = new SubmarineMetaStore(submarineConf);

  @Inject
  public MetaStoreRestApi() {
  }

  @POST
  @Path("/database/create")
  @SubmarineApi
  public Response createDatabase(@QueryParam("database") String databaseJson) {
    try {
      Database database = gson.fromJson(databaseJson, Database.class);
      submarineMetaStore.createDatabase(database);
    } catch (MetaException e) {
      LOG.error(e.getMessage(), e);
      return new JsonResponse.Builder<String>(Response.Status.INTERNAL_SERVER_ERROR)
              .success(false).build();
    } catch (JsonSyntaxException | InvalidObjectException e) {
      LOG.error(e.getMessage(), e);
      return new JsonResponse.Builder<String>(Response.Status.BAD_REQUEST)
              .success(false).build();
    }
    return new JsonResponse.Builder<String>(Response.Status.OK)
            .success(true).build();
  }

  @POST
  @Path("/table/create")
  @SubmarineApi
  public Response createTable(@QueryParam("database") String tableJson) {
    try {
      Table table = gson.fromJson(tableJson, Table.class);
      submarineMetaStore.createTable(table);
    } catch (MetaException e) {
      LOG.error(e.getMessage(), e);
      return new JsonResponse.Builder<String>(Response.Status.INTERNAL_SERVER_ERROR)
              .success(false).build();
    } catch (JsonSyntaxException | InvalidObjectException e) {
      LOG.error(e.getMessage(), e);
      return new JsonResponse.Builder<String>(Response.Status.BAD_REQUEST)
              .success(false).build();
    }
    return new JsonResponse.Builder<String>(Response.Status.OK)
            .success(true).build();
  }

  @GET
  @Path("/database/count")
  @SubmarineApi
  public Response getDatabaseCount() {
    int databaseCount = -1;
    try {
      databaseCount = submarineMetaStore.getDatabaseCount();
    } catch (MetaException e) {
      LOG.error(e.getMessage(), e);
      return new JsonResponse.Builder<Integer>(
              Response.Status.INTERNAL_SERVER_ERROR).success(false)
              .result(databaseCount).build();
    }
    return new JsonResponse.Builder<Integer>(Response.Status.OK)
            .success(true).result(databaseCount).build();
  }

  @GET
  @Path("/database/list")
  @SubmarineApi
  public Response listDatabases() {
    List<String> allDatabases;
    try {
      allDatabases = submarineMetaStore.getAllDatabases();
    } catch (MetaException e) {
      LOG.error(e.getMessage(), e);
      return new JsonResponse.Builder<String>(Response.Status.INTERNAL_SERVER_ERROR)
              .success(false).build();
    }
    return new JsonResponse.Builder<List<String>>(Response.Status.OK)
            .success(true).result(allDatabases).build();
  }

  @GET
  @Path("/database")
  @SubmarineApi
  public Response getDatabase(@QueryParam("database") String name) {
    Database database;
    try {
      database = submarineMetaStore.getDatabase(name);
    } catch (NoSuchObjectException e) {
      LOG.error(e.getMessage(), e);
      return new JsonResponse.Builder<String>(Response.Status.NOT_FOUND)
              .success(false).build();
    }
    return new JsonResponse.Builder<Database>(Response.Status.OK)
            .success(true).result(database).build();
  }

  @DELETE
  @Path("/database")
  @SubmarineApi
  public Response dropDatabase(@QueryParam("database") String name) {
    try {
      submarineMetaStore.dropDatabase(name);
    } catch (MetaException e) {
      LOG.error(e.getMessage(), e);
      return new JsonResponse.Builder<String>(Response.Status.INTERNAL_SERVER_ERROR)
              .success(false).build();
    } catch (NoSuchObjectException e) {
      LOG.error(e.getMessage(), e);
      return new JsonResponse.Builder<String>(Response.Status.NOT_FOUND)
              .success(false).build();
    }
    return new JsonResponse.Builder<String>(Response.Status.OK)
            .success(true).build();
  }

  @GET
  @Path("/table/list")
  @SubmarineApi
  public Response listTables(@QueryParam("database") String databaseName) {
    List<String> tables;
    try {
      tables = submarineMetaStore.getAllTables(databaseName);
    } catch (MetaException e) {
      LOG.error(e.getMessage(), e);
      return new JsonResponse.Builder<String>(Response.Status.INTERNAL_SERVER_ERROR)
              .success(false).build();
    }
    return new JsonResponse.Builder<List<String>>(Response.Status.OK)
            .success(true).result(tables).build();
  }

  @GET
  @Path("/table/count")
  @SubmarineApi
  public Response getTableCount() {
    int tableCount;
    try {
      tableCount = submarineMetaStore.getTableCount();
    } catch (MetaException e) {
      LOG.error(e.getMessage(), e);
      return new JsonResponse.Builder<String>(Response.Status.INTERNAL_SERVER_ERROR)
              .success(false).build();
    }
    return new JsonResponse.Builder<Integer>(Response.Status.OK)
            .success(true).result(tableCount).build();
  }

  @GET
  @Path("/table")
  @SubmarineApi
  public Response getTable(@QueryParam("database") String databaseName,
                           @QueryParam("table") String tableName) {
    Table table;
    try {
      table = submarineMetaStore.getTable(databaseName, tableName);
    } catch (MetaException e) {
      LOG.error(e.getMessage(), e);
      return new JsonResponse.Builder<String>(Response.Status.INTERNAL_SERVER_ERROR)
              .success(false).build();
    }
    return new JsonResponse.Builder<Table>(Response.Status.OK)
            .success(true).result(table).build();
  }

  @DELETE
  @Path("/table")
  @SubmarineApi
  public Response dropTable(@QueryParam("database") String databaseName,
                            @QueryParam("table") String tableName) {
    try {
      submarineMetaStore.dropTable(databaseName, tableName);
    } catch (MetaException e) {
      LOG.error(e.getMessage(), e);
      return new JsonResponse.Builder<String>(Response.Status.INTERNAL_SERVER_ERROR)
              .success(false).build();
    } catch (NoSuchObjectException e) {
      LOG.error(e.getMessage(), e);
      return new JsonResponse.Builder<String>(Response.Status.NOT_FOUND)
              .success(false).build();
    } catch (InvalidInputException | InvalidObjectException e) {
      LOG.error(e.getMessage(), e);
      return new JsonResponse.Builder<String>(Response.Status.BAD_REQUEST)
              .success(false).build();
    }
    return new JsonResponse.Builder<String>(Response.Status.OK)
            .success(true).build();
  }
}
