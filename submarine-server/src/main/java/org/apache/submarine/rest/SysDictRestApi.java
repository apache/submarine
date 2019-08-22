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
package org.apache.submarine.rest;

import com.github.pagehelper.PageInfo;
import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import org.apache.ibatis.session.RowBounds;
import org.apache.ibatis.session.SqlSession;
import org.apache.submarine.annotation.SubmarineApi;
import org.apache.submarine.database.MyBatisUtil;
import org.apache.submarine.database.entity.QueryResult;
import org.apache.submarine.database.entity.SysDict;
import org.apache.submarine.database.mappers.SysDictMapper;
import org.apache.submarine.database.mappers.CommonMapper;
import org.apache.submarine.server.JsonResponse;
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
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Path("/sys")
@Produces("application/json")
@Singleton
public class SysDictRestApi {
  private static final Logger LOG = LoggerFactory.getLogger(SysDictRestApi.class);

  private static final Gson gson = new Gson();

  public static final String KEY_RECORDS = "records";
  public static final String KEY_TOTAL = "total";

  @Inject
  public SysDictRestApi() {
  }

  @GET
  @Path("/dict/list")
  @SubmarineApi
  public Response queryDictList(@QueryParam("dictCode") String dictCode,
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
    PageInfo<SysDict> page = new PageInfo<SysDict>(list);
    QueryResult queryResult = new QueryResult(list, page.getTotal());

    return new JsonResponse.Builder<QueryResult>(Response.Status.OK)
        .success(true).result(queryResult).build();
  }

  @POST
  @Path("/dict/add")
  @SubmarineApi
  public Response addDict(String params) {
    HashMap<String, String> mapParams
        = gson.fromJson(params, new TypeToken<HashMap<String, String>>() {}.getType());
    String dictCode = mapParams.get("dictCode");
    String dictName = mapParams.get("dictName");
    String description = mapParams.get("description");

    LOG.info("addDict dictCode:{}, dictName:{}, description:{}",
        dictCode, dictName, description);

    try {
      SysDict sysDict = new SysDict();
      sysDict.setDictCode(dictCode);
      sysDict.setDictName(dictName);
      sysDict.setDescription(description);
      sysDict.setCreateTime(new Date());

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

      return new JsonResponse.Builder<>(Response.Status.OK)
          .message("Save the dictionary successfully!").success(true).build();
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      return new JsonResponse.Builder<>(Response.Status.OK)
          .message("Saving dictionary failed!").success(false).build();
    }
  }

  @PUT
  @Path("/dict/edit")
  @SubmarineApi
  public Response editDict(SysDict sysDict) {
    try {
      SqlSession sqlSession = MyBatisUtil.getSqlSession();
      SysDictMapper sysDictMapper = sqlSession.getMapper(SysDictMapper.class);
      try {
        SysDict dict = sysDictMapper.getById(sysDict.getId());
        if (dict == null) {
          return new JsonResponse.Builder<>(Response.Status.OK)
              .message("Can not found dict:" + sysDict.getId()).success(false).build();
        }
        sysDictMapper.updateBy(sysDict);
        sqlSession.commit();
      } catch (Exception e) {
        LOG.error(e.getMessage(), e);
      } finally {
        sqlSession.close();
      }
      return new JsonResponse.Builder<>(Response.Status.OK)
          .message("Update the dictionary successfully!").success(true).build();
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      return new JsonResponse.Builder<>(Response.Status.OK)
          .message("Update dictionary failed!").success(false).build();
    }
  }

  @DELETE
  @Path("/dict/delete")
  @SubmarineApi
  public Response deleteDict(@QueryParam("id") String dictId, @QueryParam("deleted") int deleted) {
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
      return new JsonResponse.Builder<>(Response.Status.OK)
          .message(msgOperation + " the dictionary successfully!").success(true).build();
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      return new JsonResponse.Builder<>(Response.Status.OK)
          .message(msgOperation + " dictionary failed!").success(false).build();
    }
  }

  @DELETE
  @Path("/dict/remove")
  @SubmarineApi
  public Response removeDict(String dictId) {
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
      return new JsonResponse.Builder<>(Response.Status.OK)
          .message("Delete the dictionary successfully!").success(true).build();
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      return new JsonResponse.Builder<>(Response.Status.OK)
          .message("Delete dictionary failed!").success(false).build();
    }
  }

  @GET
  @Path("/dictItem/list")
  @SubmarineApi
  public Response queryDictItemList(@QueryParam("column") String column,
                                @QueryParam("field") String field,
                                @QueryParam("order") String order,
                                @QueryParam("pageNo") int pageNo,
                                @QueryParam("pageSize") int pageSize) {
    LOG.info("queryDictItemList column:{}, field:{}, order:{}, pageNo:{}, pageSize:{}",
        column, field, order, pageNo, pageSize);

    return new JsonResponse.Builder<>(Response.Status.OK).result(null).success(true).build();
  }

  @GET
  @Path("/duplicateCheck")
  @SubmarineApi
  public Response duplicateCheck(@QueryParam("tableName") String tableName,
                                 @QueryParam("fieldName") String fieldName,
                                 @QueryParam("fieldVal") String fieldVal,
                                 @QueryParam("dataId") String dataId) {
    LOG.info("tableName:{}, fieldName:{}, fieldVal:{}, dataId:{}",
        tableName, fieldName, fieldVal, dataId);

    SqlSession sqlSession = MyBatisUtil.getSqlSession();
    CommonMapper systemMapper = sqlSession.getMapper(CommonMapper.class);
    Long count = 0L;
    try {
      Map<String, Object> params = new HashMap<>();
      params.put("tableName", tableName);
      params.put("fieldName", fieldName);
      params.put("fieldVal", fieldVal);
      params.put("dataId", dataId);
      count = systemMapper.duplicateCheck(params);
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
    } finally {
      sqlSession.close();
    }

    if (count == null || count == 0) {
      LOG.info("This value is available");
      return new JsonResponse.Builder<>(Response.Status.OK)
          .message("This value is available!").success(true).build();
    } else {
      LOG.info("This value already exists is not available!");
      return new JsonResponse.Builder<>(Response.Status.OK)
          .message("This value already exists is not available!").success(false).build();
    }
  }
}
