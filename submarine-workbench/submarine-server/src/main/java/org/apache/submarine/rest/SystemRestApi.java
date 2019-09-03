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

import com.google.gson.Gson;
import org.apache.ibatis.session.SqlSession;
import org.apache.submarine.annotation.SubmarineApi;
import org.apache.submarine.database.MyBatisUtil;
import org.apache.submarine.database.mappers.SystemMapper;
import org.apache.submarine.server.JsonResponse;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.inject.Inject;
import javax.inject.Singleton;
import javax.ws.rs.GET;
import javax.ws.rs.Path;
import javax.ws.rs.Produces;
import javax.ws.rs.QueryParam;
import javax.ws.rs.core.Response;
import java.util.HashMap;
import java.util.Map;

@Path("/sys")
@Produces("application/json")
@Singleton
public class SystemRestApi {
  private static final Logger LOG = LoggerFactory.getLogger(SystemRestApi.class);

  private static final Gson gson = new Gson();

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
}
