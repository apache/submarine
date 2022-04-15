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
package org.apache.submarine.server.rest.workbench;

import com.github.pagehelper.PageInfo;
import org.apache.commons.lang.StringUtils;
import org.apache.ibatis.session.SqlSession;
import org.apache.submarine.server.rest.workbench.annotation.SubmarineApi;
import org.apache.submarine.server.database.workbench.entity.SysUserEntity;
import org.apache.submarine.server.database.workbench.mappers.SystemMapper;
import org.apache.submarine.server.database.workbench.service.SysUserService;
import org.apache.submarine.server.database.database.utils.MyBatisUtil;
import org.apache.submarine.server.utils.response.JsonResponse;
import org.apache.submarine.server.utils.response.JsonResponse.ListResult;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.inject.Inject;
import javax.inject.Singleton;
import javax.ws.rs.GET;
import javax.ws.rs.Path;
import javax.ws.rs.PathParam;
import javax.ws.rs.Produces;
import javax.ws.rs.QueryParam;
import javax.ws.rs.core.Response;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Path("/sys")
@Produces("application/json")
@Singleton
public class SystemRestApi {
  private static final Logger LOG = LoggerFactory.getLogger(SystemRestApi.class);

  private SysUserService userService = new SysUserService();

  @Inject
  public SystemRestApi() {
  }

  @GET
  @Path("/duplicateCheck")
  @SubmarineApi
  public Response duplicateCheck(@QueryParam("tableName") String tableName,
                                 @QueryParam("fieldName") String fieldName,
                                 @QueryParam("fieldVal") String fieldVal,
                                 @QueryParam("equalFieldName") String equalFieldName,
                                 @QueryParam("equalFieldVal") String equalFieldVal,
                                 @QueryParam("dataId") String dataId) {
    LOG.info("tableName:{}, fieldName:{}, fieldVal:{}, equalFieldName:{}, equalFieldVal:{}, dataId:{}",
        tableName, fieldName, fieldVal, equalFieldName, equalFieldVal, dataId);

    SqlSession sqlSession = MyBatisUtil.getSqlSession();
    SystemMapper systemMapper = sqlSession.getMapper(SystemMapper.class);
    Long count = 0L;
    try {
      Map<String, Object> params = new HashMap<>();
      params.put("tableName", tableName);
      params.put("fieldName", fieldName);
      params.put("fieldVal", fieldVal);
      params.put("equalFieldName", equalFieldName);
      params.put("equalFieldVal", equalFieldVal);
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

  @GET
  @Path("/searchSelect/{tableName}")
  @SubmarineApi
  public Response searchSelect(@PathParam("tableName") String tableName,
                               @QueryParam("keyword") String keyword) {

    if (StringUtils.equals(tableName, "sys_user")) {
      List<SysUserEntity> list = null;
      try {
        list = userService.queryPageList(keyword, null, null, null, null, 1, 1000);
      } catch (Exception e) {
        LOG.error(e.getMessage(), e);
        return new JsonResponse.Builder<>(Response.Status.OK).success(false).build();
      }
      PageInfo<SysUserEntity> page = new PageInfo<>(list);
      ListResult<SysUserEntity> listResult = new ListResult(list, page.getTotal());

      return new JsonResponse.Builder<ListResult<SysUserEntity>>(Response.Status.OK)
          .success(true).result(listResult).build();
    }

    return new JsonResponse.Builder<ListResult<SysUserEntity>>(Response.Status.OK)
        .success(false).build();
  }
}
