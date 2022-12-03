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

package org.apache.submarine.server.security.oidc;

import com.github.tomakehurst.wiremock.client.WireMock;
import com.github.tomakehurst.wiremock.junit.WireMockRule;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.reflect.TypeToken;
import org.apache.submarine.commons.utils.SubmarineConfVars;
import org.apache.submarine.commons.utils.SubmarineConfiguration;

import org.apache.submarine.server.api.environment.EnvironmentId;
import org.apache.submarine.server.api.workbench.UserInfo;
import org.apache.submarine.server.rest.workbench.SysUserRestApi;
import org.apache.submarine.server.security.SecurityFactory;
import org.apache.submarine.server.security.SecurityProvider;
import org.apache.submarine.server.security.common.RegistryUserActionAdapter;
import org.apache.submarine.server.utils.gson.EnvironmentIdDeserializer;
import org.apache.submarine.server.utils.gson.EnvironmentIdSerializer;
import org.apache.submarine.server.utils.response.JsonResponse;
import org.junit.After;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.mockito.Mockito;
import org.pac4j.core.config.Config;
import org.pac4j.core.profile.UserProfile;
import org.pac4j.core.util.Pac4jConstants;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.servlet.FilterChain;
import javax.servlet.ServletException;
import javax.servlet.http.HttpServletResponse;
import javax.ws.rs.core.Response;
import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.lang.reflect.Type;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

import static com.github.tomakehurst.wiremock.client.WireMock.aResponse;
import static com.github.tomakehurst.wiremock.client.WireMock.urlEqualTo;
import static org.apache.submarine.server.security.oidc.OidcConfig.CLIENT_ID_PROP;
import static org.apache.submarine.server.security.oidc.OidcConfig.CLIENT_SECRET_PROP;
import static org.apache.submarine.server.security.oidc.OidcConfig.DISCOVER_URI_PROP;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

public class SubmarineAuthOidcTest {

  private static final Logger LOG = LoggerFactory.getLogger(SubmarineAuthOidcTest.class);

  private static final SubmarineConfiguration conf = SubmarineConfiguration.getInstance();

  private SysUserRestApi sysUserRestApi;

  private RegistryUserActionAdapter userActionAdapter;

  private static final GsonBuilder gsonBuilder = new GsonBuilder()
          .registerTypeAdapter(EnvironmentId.class, new EnvironmentIdSerializer())
          .registerTypeAdapter(EnvironmentId.class, new EnvironmentIdDeserializer());
  private static final Gson gson = gsonBuilder.setDateFormat("yyyy-MM-dd HH:mm:ss").create();

  @Rule
  public final WireMockRule wireMockRule = new WireMockRule(8080);

  @Before
  public void before() {
    conf.updateConfiguration("submarine.auth.type", "oidc");
    conf.updateConfiguration(CLIENT_ID_PROP, "test");
    conf.updateConfiguration(CLIENT_SECRET_PROP, "secret");
    conf.updateConfiguration(DISCOVER_URI_PROP,
        "http://localhost:8080/auth/realms/test-login/.well-known/openid-configuration");
    conf.setJdbcUrl("jdbc:mysql://127.0.0.1:3306/submarine_test?" +
            "useUnicode=true&" +
            "characterEncoding=UTF-8&" +
            "autoReconnect=true&" +
            "failOverReadOnly=false&" +
            "zeroDateTimeBehavior=convertToNull&" +
            "useSSL=false");
    conf.setJdbcUserName("submarine_test");
    conf.setJdbcPassword("password_test");

    sysUserRestApi = new SysUserRestApi();
    userActionAdapter = new RegistryUserActionAdapter();

    // Add oidc mock endpoint
    // Based on the token, we currently use the following two endpoints:
    // 1. openid-configuration
    String openidConfig = getResourceFileContent("security/openid-configuration.json");
    wireMockRule.stubFor(
        WireMock.get(urlEqualTo("/auth/realms/test-login/.well-known/openid-configuration"))
            .willReturn(aResponse().withHeader("Content-Type", "application/json")
                .withBody(openidConfig)
            )
    );
    // 2. userinfo
    String userInfo = getResourceFileContent("security/user-info.json");
    wireMockRule.stubFor(
        WireMock.get(urlEqualTo("/auth/realms/test-login/protocol/openid-connect/userinfo"))
            .willReturn(aResponse().withHeader("Content-Type", "application/json")
                .withBody(userInfo)
            )
    );
  }

  public static String getResourceFileContent(String resource) {
    File file = new File(Objects.requireNonNull(
        SubmarineAuthOidcTest.class.getClassLoader().getResource(resource)).getPath()
    );
    try {
      return new String(Files.readAllBytes(Paths.get(file.toString())));
    } catch (IOException e) {
      LOG.error("Can not find file: " + resource, e);
      return null;
    }
  }

  @Test
  public void testOidcType() throws ServletException, IOException {
    // test auth type config
    String authType = conf.getString(SubmarineConfVars.ConfVars.SUBMARINE_AUTH_TYPE);
    assertEquals(authType, "oidc");

    // test provider
    Optional<SecurityProvider> providerOptional = SecurityFactory.getSecurityProvider();
    SecurityProvider provider = providerOptional.get();
    assertNotNull(provider);
    assertEquals(provider.getFilterClass(), OidcFilter.class);
    Config config = provider.getConfig();
    assertTrue(config.getClients().findClient("headerClient").isPresent());
    assertTrue(config.getClients().findClient("oidcClient").isPresent());

    // create filter involved objects
    // 1. test filter
    OidcFilter filterTest = new OidcFilter();
    filterTest.init(null);
    // 2. filter chain
    FilterChain mockFilterChain = Mockito.mock(FilterChain.class);
    // 3. http request
    MockOidcHttpServletRequest mockRequest = new MockOidcHttpServletRequest();
    mockRequest.setRequestURL(new StringBuffer("/api/sys/user/info"));
    // 4. http response
    HttpServletResponse mockResponse = Mockito.mock(HttpServletResponse.class);
    StringWriter out = new StringWriter();
    PrintWriter printOut = new PrintWriter(out);
    when(mockResponse.getWriter()).thenReturn(printOut);

    // test no header
    filterTest.doFilter(mockRequest, mockResponse, mockFilterChain);
    verify(mockResponse).sendError(HttpServletResponse.SC_UNAUTHORIZED,
        "The token/session is not valid.");

    // test header, here we use a fake Token to simulate login
    mockRequest.setHeader("Authorization", "Bearer XXX");
    filterTest.doFilter(mockRequest, mockResponse, mockFilterChain);
    verify(mockFilterChain).doFilter(mockRequest, mockResponse);
    assertNotNull(mockRequest.getAttribute(Pac4jConstants.USER_PROFILES));

    // Since we are not callback user, we can simulate creating oidc user
    Map<String, UserProfile> profiles = (Map<String, UserProfile>)
        mockRequest.getAttribute(Pac4jConstants.USER_PROFILES);
    UserProfile profile = profiles.get("HeaderClient");
    userActionAdapter.createUndefinedUser(profile);

    // test get user info
    Response response = sysUserRestApi.info(mockRequest, mockResponse);
    assertEquals(response.getStatus(), Response.Status.OK.getStatusCode());
    String entity = (String) response.getEntity();
    Type type = new TypeToken<JsonResponse<UserInfo>>() { }.getType();
    JsonResponse<UserInfo> jsonResponse = gson.fromJson(entity, type);
    assertEquals(jsonResponse.getResult().getName(), "oidc_test");
  }

  @After
  public void after() {
    conf.updateConfiguration("submarine.auth.type", "none");
  }
}
