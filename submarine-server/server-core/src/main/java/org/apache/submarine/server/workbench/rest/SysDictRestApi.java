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

import com.github.pagehelper.PageInfo;
import com.google.gson.Gson;
import org.apache.ibatis.session.RowBounds;
import org.apache.ibatis.session.SqlSession;
import org.apache.submarine.server.workbench.annotation.SubmarineApi;
import org.apache.submarine.server.workbench.database.MyBatisUtil;
import org.apache.submarine.server.workbench.database.entity.SysDict;
import org.apache.submarine.server.workbench.database.mappers.SysDictMapper;
import org.apache.submarine.server.workbench.server.JsonResponse;
import org.apache.submarine.server.workbench.server.JsonResponse.ListResult;
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
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Path("/sys/dict")
@Produces("application/json")
@Singleton
public class SysDictRestApi {
  private static final Logger LOG = LoggerFactory.getLogger(SysDictRestApi.class);

  private static final Gson gson = new Gson();

  @Inject
  public SysDictRestApi() {
  }

  @GET
  @Path("/list")
  @SubmarineApi
  public Response list(@QueryParam("dictCode") String dictCode,
                       @QueryParam("dictName") String dictName,
                       @QueryParam("column") String column,
                       @QueryParam("field") String field,
                       @QueryParam("order") String order,
                       @QueryParam("pageNo") int pageNo,
                       @QueryParam("pageSize") int pageSize) {
    LOG.info("queryDictList column:{}, field:{}, order:{}, pageNo:{}, pageSize:{}",
        column, field, order, pageNo, pageSize);

    List<SysDict> list = null;
    SqlSession sqlSession = MyBatisUtil.getSqlSession();
    SysDictMapper sysDictMapper = sqlSession.getMapper(SysDictMapper.class);
    try {
      Map<String, Object> where = new HashMap<>();
      where.put("dictCode", dictCode);
      where.put("dictName", dictName);
      list = sysDictMapper.selectAll(where, new RowBounds(pageNo, pageSize));
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      return new JsonResponse.Builder<>(Response.Status.OK).success(false).build();
    } finally {
      sqlSession.close();
    }
    PageInfo<SysDict> page = new PageInfo<>(list);
    ListResult<SysDict> listResult = new ListResult(list, page.getTotal());

    return new JsonResponse.Builder<ListResult<SysDict>>(Response.Status.OK)
        .success(true).result(listResult).build();
  }

  @POST
  @Path("/add")
  @SubmarineApi
  public Response add(SysDict sysDict) {
    LOG.info("add Dict:{}", sysDict.toString());

    try {
      SqlSession sqlSession = MyBatisUtil.getSqlSession();
      SysDictMapper sysDictMapper = sqlSession.getMapper(SysDictMapper.class);
      try {
        sysDictMapper.insertSysDict(sysDict);
        sqlSession.commit();
      } catch (Exception e) {
        LOG.error(e.getMessage(), e);
      } finally {
        sqlSession.close();
      }
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      return new JsonResponse.Builder<>(Response.Status.OK)
          .message("Saving dictionary failed!").success(false).build();
    }

    return new JsonResponse.Builder<>(Response.Status.OK)
        .message("Save dictionary successfully!").success(true).build();
  }

  @PUT
  @Path("/edit")
  @SubmarineApi
  public Response edit(SysDict sysDict) {
    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      SysDictMapper sysDictMapper = sqlSession.getMapper(SysDictMapper.class);
      SysDict dict = sysDictMapper.getById(sysDict.getId());
      if (dict == null) {
        return new JsonResponse.Builder<>(Response.Status.OK)
            .message("Can not found dict:" + sysDict.getId()).success(false).build();
      }
      sysDictMapper.updateBy(sysDict);
      sqlSession.commit();
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      return new JsonResponse.Builder<>(Response.Status.OK)
          .message("Update dictionary failed!").success(false).build();
    }

    return new JsonResponse.Builder<>(Response.Status.OK)
        .message("Update the dictionary successfully!").success(true).build();
  }

  @DELETE
  @Path("/delete")
  @SubmarineApi
  public Response delete(@QueryParam("id") String dictId, @QueryParam("deleted") int deleted) {
    String msgOperation = "Delete";
    if (deleted == 0) {
      msgOperation = "Restore";
    }

    try {
      SqlSession sqlSession = MyBatisUtil.getSqlSession();
      SysDictMapper sysDictMapper = sqlSession.getMapper(SysDictMapper.class);
      try {
        SysDict dict = new SysDict();
        dict.setId(dictId);
        dict.setDeleted(deleted);
        sysDictMapper.updateBy(dict);
        sqlSession.commit();
      } catch (Exception e) {
        LOG.error(e.getMessage(), e);
      } finally {
        sqlSession.close();
      }
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      return new JsonResponse.Builder<>(Response.Status.OK)
          .message(msgOperation + " dictionary failed!").success(false).build();
    }

    return new JsonResponse.Builder<>(Response.Status.OK)
        .message(msgOperation + " the dictionary successfully!").success(true).build();
  }

  @DELETE
  @Path("/remove")
  @SubmarineApi
  public Response remove(String dictId) {
    try {
      SqlSession sqlSession = MyBatisUtil.getSqlSession();
      SysDictMapper sysDictMapper = sqlSession.getMapper(SysDictMapper.class);
      try {
        sysDictMapper.deleteById(dictId);
        sqlSession.commit();
      } catch (Exception e) {
        LOG.error(e.getMessage(), e);
      } finally {
        sqlSession.close();
      }
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      return new JsonResponse.Builder<>(Response.Status.OK)
          .message("Delete dictionary failed!").success(false).build();
    }

    return new JsonResponse.Builder<>(Response.Status.OK)
        .message("Delete the dictionary successfully!").success(true).build();
  }
}
