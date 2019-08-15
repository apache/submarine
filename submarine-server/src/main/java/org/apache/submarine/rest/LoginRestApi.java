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

import com.google.gson.Gson;
import org.apache.submarine.annotation.SubmarineApi;
import org.apache.submarine.entity.User;
import org.apache.submarine.server.JsonResponse;

import javax.inject.Inject;
import javax.inject.Singleton;
import javax.ws.rs.POST;
import javax.ws.rs.Path;
import javax.ws.rs.Produces;
import javax.ws.rs.core.MediaType;
import javax.ws.rs.core.Response;

@Path("/auth")
@Produces("application/json")
@Singleton
public class LoginRestApi {
  private static final Gson gson = new Gson();

  @Inject
  public LoginRestApi() {
  }

  @POST
  @Path("/login")
  @SubmarineApi
  public Response login(String loginParams) {
    User.Builder userBuilder = new User.Builder("4291d7da9005377ec9aec4a71ea837f", "liuxun");
    User user = userBuilder.avatar("https://gw.alipayobjects.com/zos/rmsportal/jZUIxmJycoymBprLOUbT.png")
        .status(1).telephone("").lastLoginIp("27.154.74.117")
        .lastLoginTime(1534837621348L).creatorId("admin").createTime(1497160610259L)
        .deleted(0).roleId("admin").lang("zh-CN")
        .token("4291d7da9005377ec9aec4a71ea837f").build();

    return new JsonResponse<>(Response.Status.OK, "", user).build();
  }

  @POST
  @Path("/2step-code")
  @SubmarineApi
  public Response step() {
    String data = "{stepCode:1}";

    // return new JsonResponse<>(Response.Status.OK, "", json).build();
    return Response.ok().status(Response.Status.OK)
        .type(MediaType.APPLICATION_JSON).entity(data).build();
  }
}
