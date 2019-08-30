/**
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * <p>
 * http://www.apache.org/licenses/LICENSE-2.0
 * <p>
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. See accompanying LICENSE file.
 */
package org.apache.submarine.rest;

import com.github.pagehelper.PageInfo;
import com.google.gson.Gson;
import org.apache.ibatis.session.RowBounds;
import org.apache.ibatis.session.SqlSession;
import org.apache.submarine.annotation.SubmarineApi;
import org.apache.submarine.database.MyBatisUtil;
import org.apache.submarine.database.entity.QueryResult;
import org.apache.submarine.database.entity.SysDictItem;
import org.apache.submarine.database.mappers.SysDictItemMapper;
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
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Path("/sys/dictItem")
@Produces("application/json")
@Singleton
public class SysDictItemRestApi {
  private static final Logger LOG = LoggerFactory.getLogger(SysDictItemRestApi.class);

  private static final Gson gson = new Gson();

  @Inject
  public SysDictItemRestApi() {
  }

  @GET
  @Path("/list")
  @SubmarineApi
  public Response list(@QueryParam("dictCode") String dictCode,
                       @QueryParam("itemText") String itemText,
                       @QueryParam("itemValue") String itemValue,
                       @QueryParam("column") String column,
                       @QueryParam("field") String field,
                       @QueryParam("order") String order,
                       @QueryParam("pageNo") int pageNo,
                       @QueryParam("pageSize") int pageSize) {
    LOG.info("queryList dictId:{}, itemText:{}, itemValue:{}, pageNo:{}, pageSize:{}",
        dictCode, itemText, itemValue, pageNo, pageSize);

    List<SysDictItem> list = null;
    SqlSession sqlSession = MyBatisUtil.getSqlSession();
    SysDictItemMapper sysDictItemMapper = sqlSession.getMapper(SysDictItemMapper.class);
    try {
      Map<String, Object> where = new HashMap<>();
      where.put("dictCode", dictCode);
      where.put("itemText", itemText);
      where.put("itemValue", itemValue);
      list = sysDictItemMapper.selectAll(where, new RowBounds(pageNo, pageSize));
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      return new JsonResponse.Builder<>(Response.Status.OK).success(false).build();
    } finally {
      sqlSession.close();
    }
    PageInfo<SysDictItem> page = new PageInfo<>(list);
    QueryResult<SysDictItem> queryResult = new QueryResult(list, page.getTotal());

    return new JsonResponse.Builder<QueryResult>(Response.Status.OK)
        .success(true).result(queryResult).build();
  }

  @POST
  @Path("/add")
  @SubmarineApi
  public Response add(SysDictItem sysDictItem) {
    LOG.info("addDict sysDictItem:{}", sysDictItem.toString());

    try {
      SqlSession sqlSession = MyBatisUtil.getSqlSession();
      SysDictItemMapper sysDictItemMapper = sqlSession.getMapper(SysDictItemMapper.class);
      try {
        sysDictItemMapper.insertSysDictItem(sysDictItem);
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
  @Path("/edit")
  @SubmarineApi
  public Response edit(SysDictItem sysDictItem) {
    try {
      SqlSession sqlSession = MyBatisUtil.getSqlSession();
      SysDictItemMapper sysDictItemMapper = sqlSession.getMapper(SysDictItemMapper.class);
      try {
        SysDictItem dictItem = sysDictItemMapper.getById(sysDictItem.getId());
        if (dictItem == null) {
          return new JsonResponse.Builder<>(Response.Status.OK)
              .message("Can not found dict item:" + sysDictItem.getId()).success(false).build();
        }
        sysDictItemMapper.updateBy(sysDictItem);
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
  @Path("/delete")
  @SubmarineApi
  public Response delete(@QueryParam("id") String id, @QueryParam("deleted") int deleted) {
    String msgOperation = "Delete";
    if (deleted == 0) {
      msgOperation = "Restore";
    }

    try {
      SqlSession sqlSession = MyBatisUtil.getSqlSession();
      SysDictItemMapper sysDictItemMapper = sqlSession.getMapper(SysDictItemMapper.class);
      try {
        SysDictItem dictItem = new SysDictItem();
        dictItem.setId(id);
        dictItem.setDeleted(deleted);
        sysDictItemMapper.updateBy(dictItem);
        sqlSession.commit();
      } catch (Exception e) {
        LOG.error(e.getMessage(), e);
      } finally {
        sqlSession.close();
      }
      return new JsonResponse.Builder<>(Response.Status.OK)
          .message(msgOperation + " the dict item successfully!").success(true).build();
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      return new JsonResponse.Builder<>(Response.Status.OK)
          .message(msgOperation + " dict item failed!").success(false).build();
    }
  }

  @DELETE
  @Path("/remove")
  @SubmarineApi
  public Response remove(String id) {
    try {
      SqlSession sqlSession = MyBatisUtil.getSqlSession();
      SysDictItemMapper sysDictItemMapper = sqlSession.getMapper(SysDictItemMapper.class);
      try {
        sysDictItemMapper.deleteById(id);
        sqlSession.commit();
      } catch (Exception e) {
        LOG.error(e.getMessage(), e);
      } finally {
        sqlSession.close();
      }
      return new JsonResponse.Builder<>(Response.Status.OK)
          .message("Delete the dict item successfully!").success(true).build();
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      return new JsonResponse.Builder<>(Response.Status.OK)
          .message("Delete dict item failed!").success(false).build();
    }
  }
}
