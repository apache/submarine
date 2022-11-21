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

package org.apache.submarine.server.security.common;

import org.apache.commons.lang3.StringUtils;
import org.apache.submarine.server.rest.workbench.annotation.NoneAuth;
import org.pac4j.core.context.JEEContext;
import org.pac4j.core.context.session.JEESessionStore;
import org.pac4j.core.context.session.SessionStore;
import org.pac4j.core.engine.DefaultCallbackLogic;
import org.pac4j.core.engine.DefaultLogoutLogic;
import org.pac4j.core.engine.DefaultSecurityLogic;
import org.pac4j.core.http.adapter.HttpActionAdapter;
import org.pac4j.core.http.adapter.JEEHttpActionAdapter;
import org.pac4j.core.profile.CommonProfile;
import org.pac4j.core.profile.UserProfile;
import org.reflections.Reflections;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.servlet.FilterConfig;
import javax.servlet.ServletException;
import javax.servlet.http.HttpServletRequest;
import javax.ws.rs.DELETE;
import javax.ws.rs.GET;
import javax.ws.rs.HEAD;
import javax.ws.rs.OPTIONS;
import javax.ws.rs.POST;
import javax.ws.rs.Path;
import javax.ws.rs.PATCH;
import javax.ws.rs.PUT;
import java.lang.annotation.Annotation;
import java.lang.reflect.Method;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;
import java.util.regex.Pattern;
import java.util.stream.Stream;

import static org.reflections.scanners.Scanners.SubTypes;
import static org.reflections.scanners.Scanners.TypesAnnotated;


public class CommonFilter {

  public static final HttpActionAdapter DEFAULT_HTTP_ACTION_ADAPTER = JEEHttpActionAdapter.INSTANCE;

  public static final DefaultCallbackLogic<CommonProfile, JEEContext> CALLBACK_LOGIC =
          new DefaultCallbackLogic<>();

  public static final DefaultSecurityLogic<UserProfile, JEEContext> SECURITY_LOGIC =
          new DefaultSecurityLogic<>();

  public static final DefaultLogoutLogic<UserProfile, JEEContext> LOGOUT_LOGIC = new DefaultLogoutLogic<>();

  public static final SessionStore<JEEContext> SESSION_STORE = new JEESessionStore();

  private static final Logger LOG = LoggerFactory.getLogger(CommonFilter.class);

  /* Supported http method */
  protected final Set<Class<? extends Annotation>> SUPPORT_HTTP_METHODS =
      new HashSet<Class<? extends Annotation>>() {{
        add(GET.class);
        add(PUT.class);
        add(POST.class);
        add(DELETE.class);
        add(PATCH.class);
        add(OPTIONS.class);
        add(HEAD.class);
      }};

  /* api with the full path */
  protected final Set<String> REST_API_PATHS = new HashSet<>(16);
  /* api with the regrex path */
  protected final Set<String> REST_REGREX_API_PATHS = new HashSet<>(16);

  /**
   * Filter init
   */
  public void init(FilterConfig filterConfig) throws ServletException {
    // Scan rest api class by annotations @Path
    Reflections reflections = new Reflections("org.apache.submarine.server.rest");
    Set<Class<?>> rests = reflections.get(SubTypes.of(TypesAnnotated.with(Path.class)).asClass());
    for (Class<?> rest : rests) {
      // get path
      Path pathAnno = rest.getAnnotation(Path.class);
      String path = pathAnno.value();
      if (path.startsWith("/")) path = path.substring(1);
      if (path.endsWith("/")) path = path.substring(0, path.length() - 1);
      // loop method annotations
      Method[] methods = rest.getDeclaredMethods();
      for (Method method : methods) {
        addSupportedApiPath(path, method);
      }
    }
    LOG.info("Get security filter rest api path = {} and regrex api path = {}",
        REST_API_PATHS, REST_REGREX_API_PATHS);
  }

  /**
   * Add supported api path
   */
  private void addSupportedApiPath(String path, Method method) {
    Stream<Annotation> annotations = Arrays.stream(method.getAnnotations());
    // Only methods marked as REST http method
    if (annotations.anyMatch(annotation -> SUPPORT_HTTP_METHODS.contains(annotation.annotationType()))) {
      // Methods with the @NoneAuth require no authentication
      if (method.getAnnotation(NoneAuth.class) != null) return;
      Path pathAnno = method.getAnnotation(Path.class);
      String endpoint = pathAnno == null ? "" : pathAnno.value();

      // If endpoint is empty, the api is used as the path
      if ("".equals(endpoint) || "/".equals(endpoint)) {
        REST_API_PATHS.add(String.format("/api/%s", path));
      } else {
        if (endpoint.startsWith("/")) endpoint = endpoint.substring(1);
        if (endpoint.endsWith("/")) endpoint = endpoint.substring(0, endpoint.length() - 1);
        String api = String.format("/api/%s/%s", path, endpoint);
        if (api.matches("(.*)\\{[a-zA-Z0-9]+\\}(.*)")) {
          REST_REGREX_API_PATHS.add(api.replaceAll("\\{[a-zA-Z0-9]+\\}", "((?!\\/).)*"));
        } else {
          REST_API_PATHS.add(api);
        }
      }
    }
  }

  /**
   * Check if uri is in the list of known apis
   */
  private boolean isSupportedRest(String uri) {
    // Return true if found in the full path
    if (REST_API_PATHS.contains(uri)) return true;
    // Otherwise, do a match on the regrex path
    for (String api : REST_REGREX_API_PATHS) {
      if (Pattern.matches(api, uri)) {
        return true;
      }
    }
    return false;
  }

  /**
   * Check whether the endpoint requires authorization verification
   */
  protected boolean isProtectedApi(HttpServletRequest httpServletRequest) {
    // If it is called by python, temporarily passed
    String agentHeader = httpServletRequest.getHeader(CommonConfig.AGENT_HEADER);
    if (StringUtils.isNoneBlank(agentHeader) && CommonConfig.PYTHON_USER_AGENT.equals(agentHeader)) {
      return false;
    }
    // Now we just verify the api
    return isSupportedRest(httpServletRequest.getRequestURI());
  }
}
