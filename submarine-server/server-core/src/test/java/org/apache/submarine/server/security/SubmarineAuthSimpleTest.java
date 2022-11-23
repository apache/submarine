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

package org.apache.submarine.server.security;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.reflect.TypeToken;
import org.apache.submarine.commons.utils.SubmarineConfVars;
import org.apache.submarine.commons.utils.SubmarineConfiguration;
import org.apache.submarine.server.api.environment.EnvironmentId;
import org.apache.submarine.server.database.workbench.entity.SysUserEntity;
import org.apache.submarine.server.rest.workbench.LoginRestApi;
import org.apache.submarine.server.rest.workbench.SysUserRestApi;
import org.apache.submarine.server.security.simple.SimpleFilter;
import org.apache.submarine.server.utils.gson.EnvironmentIdDeserializer;
import org.apache.submarine.server.utils.gson.EnvironmentIdSerializer;
import org.apache.submarine.server.utils.response.JsonResponse;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.mockito.Mockito;
import org.pac4j.core.config.Config;
import org.pac4j.core.util.Pac4jConstants;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.servlet.FilterChain;
import javax.servlet.ServletException;
import javax.servlet.http.HttpServletResponse;
import javax.ws.rs.core.Response;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.lang.reflect.Type;
import java.util.Optional;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

public class SubmarineAuthSimpleTest {

  private static final SubmarineConfiguration conf = SubmarineConfiguration.getInstance();

  private static final GsonBuilder gsonBuilder = new GsonBuilder()
          .registerTypeAdapter(EnvironmentId.class, new EnvironmentIdSerializer())
          .registerTypeAdapter(EnvironmentId.class, new EnvironmentIdDeserializer());
  private static final Gson gson = gsonBuilder.setDateFormat("yyyy-MM-dd HH:mm:ss").create();

  private static final Logger LOG = LoggerFactory.getLogger(SubmarineAuthSimpleTest.class);

  private static LoginRestApi loginRestApi;
  private static SysUserRestApi sysUserRestApi;

  @Before
  public void before() {
    conf.updateConfiguration("submarine.auth.type", "simple");
    conf.setJdbcUrl("jdbc:mysql://127.0.0.1:3306/submarine_test?" +
            "useUnicode=true&" +
            "characterEncoding=UTF-8&" +
            "autoReconnect=true&" +
            "failOverReadOnly=false&" +
            "zeroDateTimeBehavior=convertToNull&" +
            "useSSL=false");
    conf.setJdbcUserName("submarine_test");
    conf.setJdbcPassword("password_test");
    loginRestApi = new LoginRestApi();
    // add a test user
    sysUserRestApi = new SysUserRestApi();
    SysUserEntity user = new SysUserEntity();
    user.setUserName("test");
    user.setRealName("test");
    user.setPassword("test");
    user.setDeleted(0);
    sysUserRestApi.add(user);
  }

  @Test
  public void testSimpleType() throws ServletException, IOException {
    // test auth type config
    String authType = conf.getString(SubmarineConfVars.ConfVars.SUBMARINE_AUTH_TYPE);
    assertEquals(authType, "simple");

    // test provider
    Optional<SecurityProvider> providerOptional = SecurityFactory.getSecurityProvider();
    SecurityProvider provider = providerOptional.get();
    assertNotNull(provider);
    assertEquals(provider.getFilterClass(), SimpleFilter.class);
    Config config = provider.getConfig();
    assertTrue(config.getClients().findClient("headerClient").isPresent());

    // test login api
    String testUsrJson = "{\"username\":\"test\",\"password\":\"test\"}";
    Response loginResp = loginRestApi.login(testUsrJson);
    assertEquals(loginResp.getStatus(), Response.Status.OK.getStatusCode());
    String entity = (String) loginResp.getEntity();
    Type type = new TypeToken<JsonResponse<SysUserEntity>>() { }.getType();
    JsonResponse<SysUserEntity> jsonResponse = gson.fromJson(entity, type);
    String token = jsonResponse.getResult().getToken();
    LOG.info("Get user token: " + token);

    // create filter involved objects
    // 1. test filter
    SimpleFilter filterTest = new SimpleFilter();
    filterTest.init(null);
    // 2. filter chain
    FilterChain mockFilterChain = Mockito.mock(FilterChain.class);
    // 3. http request
    MockHttpServletRequest mockRequest = new MockHttpServletRequest();
    mockRequest.setRequestURL(new StringBuffer("/api/sys/user/info"));
    // 4. http response
    HttpServletResponse mockResponse = Mockito.mock(HttpServletResponse.class);
    StringWriter out = new StringWriter();
    PrintWriter printOut = new PrintWriter(out);
    when(mockResponse.getWriter()).thenReturn(printOut);

    // test no header
    filterTest.doFilter(mockRequest, mockResponse, mockFilterChain);
    verify(mockResponse).sendError(HttpServletResponse.SC_UNAUTHORIZED, "The token is not valid.");

    // test header
    mockRequest.setHeader("Authorization", "Bearer " + token);
    filterTest.doFilter(mockRequest, mockResponse, mockFilterChain);
    verify(mockFilterChain).doFilter(mockRequest, mockResponse);
    assertNotNull(mockRequest.getAttribute(Pac4jConstants.USER_PROFILES));
  }

  @After
  public void after() {
    conf.updateConfiguration("submarine.auth.type", "none");
  }

}
