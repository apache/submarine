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

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import io.swagger.v3.oas.annotations.Hidden;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import org.apache.ibatis.session.SqlSession;
import org.apache.submarine.server.api.workbench.LoginParam;
import org.apache.submarine.server.rest.workbench.annotation.NoneAuth;
import org.apache.submarine.server.rest.workbench.annotation.SubmarineApi;
import org.apache.submarine.server.database.workbench.entity.SysUserEntity;
import org.apache.submarine.server.database.workbench.mappers.SysUserMapper;
import org.apache.submarine.server.database.utils.MyBatisUtil;
import org.apache.submarine.server.security.common.CommonConfig;
import org.apache.submarine.server.security.simple.SimpleLoginConfig;
import org.apache.submarine.server.utils.response.JsonResponse;
import org.mlflow_project.apachecommons.codec.digest.DigestUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.inject.Inject;
import javax.inject.Singleton;
import javax.ws.rs.Consumes;
import javax.ws.rs.POST;
import javax.ws.rs.Path;
import javax.ws.rs.Produces;
import javax.ws.rs.core.MediaType;
import javax.ws.rs.core.Response;
import java.util.Date;
import java.util.HashMap;
import java.util.Map;

@Path("/auth")
@Produces("application/json")
@Singleton
public class LoginRestApi {
  private static final Logger LOG = LoggerFactory.getLogger(LoginRestApi.class);
  private static final Gson gson = new Gson();

  @Inject
  public LoginRestApi() {
  }

  @POST
  @Path("/login")
  @SubmarineApi
  @NoneAuth
  @Hidden
  public Response login(String loginParams) {
    HashMap<String, String> mapParams
        = gson.fromJson(loginParams, new TypeToken<HashMap<String, String>>() {}.getType());

    SysUserEntity sysUser = null;
    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      SysUserMapper sysUserMapper = sqlSession.getMapper(SysUserMapper.class);
      sysUser = sysUserMapper.login(mapParams);
      if (sysUser != null) {
        // TODO(cdmikechen) By default the simple token is used,
        //  in other cases such as ldap it may need to be returned as an interface
        String token = createToken(sysUser);
        sysUser.setToken(token);
      } else {
        LOG.warn("Can not find user {}", mapParams);
        return new JsonResponse.Builder<>(Response.Status.UNAUTHORIZED)
            .message("User Not Found. Please try again!")
            .success(false)
            .build();
      }
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      return new JsonResponse.Builder<>(Response.Status.OK)
          .message(e.getMessage())
          .success(false)
          .build();
    }

    return new JsonResponse.Builder<SysUserEntity>(Response.Status.OK)
        .message("Login successfully!")
        .success(true)
        .result(sysUser)
        .build();
  }

  /**
   * Create token with user info
   */
  private String createToken(SysUserEntity sysUser) {
    HashMap<String, Object> claimsMap = new HashMap<>();
    claimsMap.put("username", sysUser.getUserName());
    claimsMap.put("realName", sysUser.getRealName());
    claimsMap.put("password", sysUser.getPassword());
    claimsMap.put("avatar", sysUser.getAvatar());
    claimsMap.put("sex", sysUser.getSex());
    claimsMap.put("status", sysUser.getStatus());
    claimsMap.put("phone", sysUser.getPhone());
    claimsMap.put("email", sysUser.getEmail());
    claimsMap.put("deptCode", sysUser.getDeptCode());
    claimsMap.put("deptName", sysUser.getDeptName());
    claimsMap.put("roleCode", sysUser.getRoleCode());
    claimsMap.put("birthday", sysUser.getBirthday());
    claimsMap.put("iat", new Date().getTime());
    claimsMap.put("exp", new Date().getTime() + CommonConfig.MAX_AGE);
    claimsMap.put("sub", "submarine");
    claimsMap.put("jti", sysUser.getId());
    return SimpleLoginConfig.getJwtGenerator().generate(claimsMap);
  }

  @POST
  @Path("/token")
  @SubmarineApi
  @NoneAuth
  @Consumes(MediaType.APPLICATION_JSON)
  @Operation(summary = "Get submarine user token",
          description = "Return user token by username and password",
          tags = {"auth"},
          responses = {
                  @ApiResponse(responseCode = "200", description = "successfully get token",
                          content = @Content(schema = @Schema(implementation = String.class)))})
  public Response token(LoginParam loginParams) {
    String token = null;
    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      SysUserMapper sysUserMapper = sqlSession.getMapper(SysUserMapper.class);
      SysUserEntity sysUser = sysUserMapper.login(Map.of(
          "username", loginParams.getUsername(),
          "password", DigestUtils.md5Hex(loginParams.getPassword()))
      );
      if (sysUser != null) {
        token = createToken(sysUser);
      } else {
        LOG.warn("Can not find user {}", loginParams.getUsername());
        return new JsonResponse.Builder<>(Response.Status.UNAUTHORIZED)
                .message("User Not Found. Please try again!")
                .success(false)
                .build();
      }
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      return new JsonResponse.Builder<>(Response.Status.OK)
          .message(e.getMessage())
          .success(false)
          .build();
    }

    return new JsonResponse.Builder<String>(Response.Status.OK)
        .message("Get token successfully!")
        .success(true)
        .result(token)
        .build();
  }

  @POST
  @Path("/2step-code")
  @SubmarineApi
  @Hidden
  public Response step() {
    String data = "{stepCode:1}";
    return new JsonResponse.Builder<String>(Response.Status.OK).success(true).result(data).build();
  }

  @POST
  @Path("/logout")
  @SubmarineApi
  @Hidden
  public Response logout() {
    return new JsonResponse.Builder<Boolean>(Response.Status.OK).success(true).result(true).build();
  }
}
