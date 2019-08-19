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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.inject.Inject;
import javax.inject.Singleton;
import javax.ws.rs.GET;
import javax.ws.rs.Path;
import javax.ws.rs.Produces;
import javax.ws.rs.QueryParam;
import javax.ws.rs.core.MediaType;
import javax.ws.rs.core.Response;

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
  @Path("/dict/list")
  @SubmarineApi
  public Response queryDictList(@QueryParam("column") String column,
                                @QueryParam("field") String field,
                                @QueryParam("order") String order,
                                @QueryParam("pageNo") int pageNo,
                                @QueryParam("pageSize") int pageSize) {
    LOG.info("queryDictList column:{}, field:{}, order:{}, pageNo:{}, pageSize:{}",
        column, field, order, pageNo, pageSize);

    String data = "";

    return Response.ok().status(Response.Status.OK).type(MediaType.APPLICATION_JSON).entity(data).build();
  }
}
