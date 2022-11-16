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

import org.apache.submarine.server.security.SecurityFactory;
import org.apache.submarine.server.security.SecurityProvider;
import org.apache.submarine.server.utils.response.JsonResponse;
import org.apache.submarine.server.utils.response.JsonResponse.ListResult;
import org.apache.submarine.server.rest.workbench.annotation.SubmarineApi;
import org.apache.submarine.server.database.workbench.entity.SysUserEntity;
import org.apache.submarine.server.database.workbench.service.SysUserService;
import org.apache.submarine.server.api.workbench.Action;
import org.apache.submarine.server.api.workbench.Permission;
import org.apache.submarine.server.api.workbench.Role;
import org.apache.submarine.server.api.workbench.UserInfo;
import org.pac4j.core.profile.CommonProfile;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.inject.Inject;
import javax.inject.Singleton;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import javax.ws.rs.DELETE;
import javax.ws.rs.GET;
import javax.ws.rs.POST;
import javax.ws.rs.PUT;
import javax.ws.rs.Path;
import javax.ws.rs.Produces;
import javax.ws.rs.QueryParam;
import javax.ws.rs.core.Response;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

@Path("/sys/user")
@Produces("application/json")
@Singleton
public class SysUserRestApi {
  private static final Logger LOG = LoggerFactory.getLogger(SysUserRestApi.class);

  private final SysUserService userService = new SysUserService();

  // default user is admin
  public static final String DEFAULT_ADMIN_UID = "e9ca23d68d884d4ebb19d07889727dae";
  // default password is `password` by angular markAsDirty method
  public static final String DEFAULT_CREATE_USER_PASSWORD = "5f4dcc3b5aa765d61d8327deb882cf99";

  @Inject
  public SysUserRestApi() {
  }

  @GET
  @Path("/list")
  @SubmarineApi
  public Response queryPageList(@QueryParam("userName") String userName,
                                @QueryParam("email") String email,
                                @QueryParam("deptCode") String deptCode,
                                @QueryParam("column") String column,
                                @QueryParam("field") String field,
                                @QueryParam("pageNo") int pageNo,
                                @QueryParam("pageSize") int pageSize) {
    LOG.debug("queryDictList userName:{}, email:{}, deptCode:{}, " +
            "column:{}, field:{}, pageNo:{}, pageSize:{}",
        userName, email, deptCode, column, field, pageNo, pageSize);

    List<SysUserEntity> list = null;
    try {
      list = userService.queryPageList(userName, email, deptCode, column, field, pageNo, pageSize);
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      return new JsonResponse.Builder<>(Response.Status.OK).success(false).build();
    }
    PageInfo<SysUserEntity> page = new PageInfo<>(list);
    ListResult<SysUserEntity> listResult = new ListResult(list, page.getTotal());

    return new JsonResponse.Builder<ListResult<SysUserEntity>>(Response.Status.OK)
        .success(true).result(listResult).build();
  }

  @PUT
  @Path("/edit")
  @SubmarineApi
  public Response edit(SysUserEntity sysUser) {
    LOG.info("edit({})", sysUser.toString());

    try {
      userService.edit(sysUser);
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      return new JsonResponse.Builder<>(Response.Status.OK)
          .message("Update user failed!").success(false).build();
    }
    return new JsonResponse.Builder<>(Response.Status.OK)
        .success(true).message("Update user successfully!").build();
  }

  @POST
  @Path("/add")
  @SubmarineApi
  public Response add(SysUserEntity sysUser) {
    LOG.info("add({})", sysUser.toString());

    try {
      userService.add(sysUser);
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      return new JsonResponse.Builder<>(Response.Status.OK).success(false)
          .message("Save user failed!").build();
    }

    return new JsonResponse.Builder<SysUserEntity>(Response.Status.OK)
        .success(true).message("Save user successfully!").result(sysUser).build();
  }

  @DELETE
  @Path("/delete")
  @SubmarineApi
  public Response delete(@QueryParam("id") String id) {
    LOG.info("delete({})", id);

    try {
      userService.delete(id);
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      return new JsonResponse.Builder<>(Response.Status.OK).success(false)
          .message("delete user failed!").build();
    }
    return new JsonResponse.Builder<>(Response.Status.OK)
        .success(true).message("delete  user successfully!").build();
  }

  @PUT
  @Path("/changePassword")
  @SubmarineApi
  public Response changePassword(SysUserEntity sysUser) {
    LOG.info("changePassword({})", sysUser.toString());

    try {
      userService.changePassword(sysUser);
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      return new JsonResponse.Builder<>(Response.Status.OK).success(false)
          .message("delete user failed!").build();
    }
    return new JsonResponse.Builder<>(Response.Status.OK)
        .success(true).message("delete  user successfully!").build();
  }

  @GET
  @Path("/info")
  @SubmarineApi
  public Response info(HttpServletRequest hsRequest, HttpServletResponse hsResponse) {
    UserInfo userInfo = null;
    // get SecurityProvider to use perform method to get user info
    Optional<SecurityProvider> securityProvider = SecurityFactory.getSecurityProvider();
    if (securityProvider.isPresent()) {
      Optional<CommonProfile> profile = securityProvider.get().perform(hsRequest, hsResponse);
      if (profile.isPresent()) {
        SysUserEntity sysUser;
        try {// find match user in db
          sysUser = userService.getUserByName(profile.get().getUsername());
        } catch (Exception e) {
          LOG.error(e.getMessage(), e);
          return new JsonResponse.Builder<>(Response.Status.OK)
                  .success(false)
                  .message("Get error when searching user name!")
                  .build();
        }
        // user not found
        if (sysUser == null || sysUser.getDeleted() == 1) {
          return new JsonResponse.Builder<>(Response.Status.OK).
                  success(false)
                  .message("User can not be found!")
                  .build();
        }
        UserInfo.Builder userInfoBuilder = new UserInfo.Builder(sysUser.getId(), sysUser.getUserName());
        userInfo = userInfoBuilder
                .username(sysUser.getUserName())
                .password("******")
                .avatar(sysUser.getAvatar())
                .status(sysUser.getStatus())
                .telephone(sysUser.getPhone())
                .lastLoginIp("******")
                .lastLoginTime(System.currentTimeMillis())
                .creatorId(sysUser.getUserName())
                .createTime(sysUser.getCreateTime().getTime())
                .merchantCode("")
                .deleted(0)
                .roleId("default")
                .role(createDefaultRole()).build();
      } else {
        return new JsonResponse.Builder<>(Response.Status.OK)
                .success(false)
                .message("User can not be found!")
                .build();
      }
    }
    if (userInfo == null) userInfo = createDefaultUser();

    return new JsonResponse.Builder<UserInfo>(Response.Status.OK)
            .success(true)
            .result(userInfo)
            .build();
  }

  /**
   * Create default role
   */
  private Role createDefaultRole() {
    // TODO(cdmikechen): Will do after the role function is completed
    List<Action> actions = new ArrayList<Action>();
    Action action1 = new Action("add", false, "add");
    Action action2 = new Action("query", false, "query");
    Action action3 = new Action("get", false, "get");
    Action action4 = new Action("update", false, "update");
    Action action5 = new Action("delete", false, "delete");
    actions.add(action1);
    actions.add(action2);
    actions.add(action3);
    actions.add(action4);
    actions.add(action5);

    Permission.Builder permissionBuilder1 = new Permission.Builder("admin", "dashboard", "dashboard");
    Permission permission1 = permissionBuilder1.actions(actions).actionEntitySet(actions).build();

    Permission.Builder permissionBuilder2 = new Permission.Builder("admin", "exception", "exception");
    Permission permission2 = permissionBuilder2.actions(actions).actionEntitySet(actions).build();

    Permission.Builder permissionBuilder3 = new Permission.Builder("admin", "result", "result");
    Permission permission3 = permissionBuilder3.actions(actions).actionEntitySet(actions).build();

    Permission.Builder permissionBuilder4 = new Permission.Builder("admin", "profile", "profile");
    Permission permission4 = permissionBuilder4.actions(actions).actionEntitySet(actions).build();

    Permission.Builder permissionBuilder5 = new Permission.Builder("admin", "table", "table");
    Permission permission5 = permissionBuilder5.actions(actions).actionEntitySet(actions).build();

    Permission.Builder permissionBuilder6 = new Permission.Builder("admin", "form", "form");
    Permission permission6 = permissionBuilder6.actions(actions).actionEntitySet(actions).build();

    Permission.Builder permissionBuilder7 = new Permission.Builder("admin", "order", "order");
    Permission permission7 = permissionBuilder7.actions(actions).actionEntitySet(actions).build();

    Permission.Builder permissionBuilder8 = new Permission.Builder("admin", "permission", "permission");
    Permission permission8 = permissionBuilder8.actions(actions).actionEntitySet(actions).build();

    Permission.Builder permissionBuilder9 = new Permission.Builder("admin", "role", "role");
    Permission permission9 = permissionBuilder9.actions(actions).actionEntitySet(actions).build();

    Permission.Builder permissionBuilder10 = new Permission.Builder("admin", "table", "table");
    Permission permission10 = permissionBuilder10.actions(actions).actionEntitySet(actions).build();

    Permission.Builder permissionBuilder11 = new Permission.Builder("admin", "user", "user");
    Permission permission11 = permissionBuilder11.actions(actions).actionEntitySet(actions).build();

    Permission.Builder permissionBuilder12 = new Permission.Builder("admin", "support", "support");
    Permission permission12 = permissionBuilder12.actions(actions).actionEntitySet(actions).build();

    List<Permission> permissions = new ArrayList<Permission>();
    permissions.add(permission1);
    permissions.add(permission2);
    permissions.add(permission3);
    permissions.add(permission4);
    permissions.add(permission5);
    permissions.add(permission6);
    permissions.add(permission7);
    permissions.add(permission8);
    permissions.add(permission9);
    permissions.add(permission10);
    permissions.add(permission11);
    permissions.add(permission12);

    Role.Builder roleBuilder = new Role.Builder("admin", "admin");
    return roleBuilder.describe("Permission")
            .status(1)
            .creatorId("system")
            .createTime(System.currentTimeMillis())
            .deleted(0)
            .permissions(permissions)
            .build();
  }

  /**
   * Create default user
   */
  private UserInfo createDefaultUser() {
    LOG.warn("Can not get user info, use a default admin user");
    UserInfo.Builder userInfoBuilder = new UserInfo.Builder(DEFAULT_ADMIN_UID, "admin");
    return userInfoBuilder.username("admin")
            .password("")
            .avatar("/avatar2.jpg")
            .status("1")
            .telephone("")
            .lastLoginIp("******")
            .lastLoginTime(System.currentTimeMillis())
            .creatorId("admin")
            .createTime(System.currentTimeMillis())
            .merchantCode("TLif2btpzg079h15bk")
            .deleted(0)
            .roleId("admin")
            .role(createDefaultRole())
            .build();
  }

  @POST
  @Path("/2step-code")
  @SubmarineApi
  public Response step() {
    String data = "{stepCode:1}";

    return new JsonResponse.Builder<>(Response.Status.OK).success(true).result(data).build();
  }
}
