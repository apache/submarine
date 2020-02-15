/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.submarine.rest;

import org.apache.commons.httpclient.Header;
import org.apache.commons.httpclient.HttpClient;
import org.apache.commons.httpclient.cookie.CookiePolicy;
import org.apache.commons.httpclient.methods.ByteArrayRequestEntity;
import org.apache.commons.httpclient.methods.DeleteMethod;
import org.apache.commons.httpclient.methods.GetMethod;
import org.apache.commons.httpclient.methods.PostMethod;
import org.apache.commons.httpclient.methods.PutMethod;
import org.apache.commons.httpclient.methods.RequestEntity;
import org.apache.commons.lang.StringUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.ws.rs.core.MediaType;
import java.io.IOException;
import java.util.regex.Pattern;

public abstract class AbstractSubmarineServerTest {
  protected static final Logger LOG = LoggerFactory.getLogger(AbstractSubmarineServerTest.class);

  static final String URL = getUrlToTest();

  public static String getUrlToTest() {
    // The submarine server deployed in k8s uses the default port of 8080 in submarine-site.xml.
    // However, port 80 is exposed through k8s Ingress.
    String url = "http://localhost:80";
    if (System.getProperty("url") != null) {
      url = System.getProperty("url");
    }
    return url;
  }

  protected static PostMethod httpPost(String path, String body) throws IOException {
    return httpPost(path, body, StringUtils.EMPTY, StringUtils.EMPTY);
  }

  protected static PostMethod httpPost(String path, String request, String user, String pwd)
      throws IOException {
    LOG.info("Connecting to {}", URL + path);

    HttpClient httpClient = new HttpClient();
    PostMethod postMethod = new PostMethod(URL + path);
    postMethod.setRequestBody(request);
    postMethod.setRequestHeader("Content-type", MediaType.APPLICATION_JSON);
    postMethod.getParams().setCookiePolicy(CookiePolicy.IGNORE_COOKIES);

    if (userAndPasswordAreNotBlank(user, pwd)) {
      postMethod.setRequestHeader("Cookie", "JSESSIONID=" + getCookie(user, pwd));
    }

    httpClient.executeMethod(postMethod);

    LOG.info("{} - {}", postMethod.getStatusCode(), postMethod.getStatusText());

    return postMethod;
  }

  protected static PutMethod httpPut(String path, String body) throws IOException {
    return httpPut(path, body, StringUtils.EMPTY, StringUtils.EMPTY);
  }

  protected static PutMethod httpPut(String path, String body, String user, String pwd) throws IOException {
    LOG.info("Connecting to {}", URL + path);
    HttpClient httpClient = new HttpClient();
    PutMethod putMethod = new PutMethod(URL + path);
    putMethod.addRequestHeader("Origin", URL);
    putMethod.setRequestHeader("Content-type", "application/yaml");
    RequestEntity entity = new ByteArrayRequestEntity(body.getBytes("UTF-8"));
    putMethod.setRequestEntity(entity);
    if (userAndPasswordAreNotBlank(user, pwd)) {
      putMethod.setRequestHeader("Cookie", "JSESSIONID=" + getCookie(user, pwd));
    }
    httpClient.executeMethod(putMethod);
    LOG.info("{} - {}", putMethod.getStatusCode(), putMethod.getStatusText());
    return putMethod;
  }

  protected static DeleteMethod httpDelete(String path) throws IOException {
    return httpDelete(path, StringUtils.EMPTY, StringUtils.EMPTY);
  }

  protected static DeleteMethod httpDelete(String path, String user, String pwd) throws IOException {
    LOG.info("Connecting to {}", URL + path);
    HttpClient httpClient = new HttpClient();
    DeleteMethod deleteMethod = new DeleteMethod(URL + path);
    deleteMethod.addRequestHeader("Origin", URL);
    if (userAndPasswordAreNotBlank(user, pwd)) {
      deleteMethod.setRequestHeader("Cookie", "JSESSIONID=" + getCookie(user, pwd));
    }
    httpClient.executeMethod(deleteMethod);
    LOG.info("{} - {}", deleteMethod.getStatusCode(), deleteMethod.getStatusText());
    return deleteMethod;
  }

  protected static GetMethod httpGet(String path) throws IOException {
    return httpGet(path, "", "");
  }

  protected static GetMethod httpGet(String path, String user, String pwd) throws IOException {
    return httpGet(path, user, pwd, "");
  }

  protected static GetMethod httpGet(String path, String user, String pwd, String cookies)
      throws IOException {
    LOG.info("Connecting to {}", URL + path);
    HttpClient httpClient = new HttpClient();
    GetMethod getMethod = new GetMethod(URL + path);
    getMethod.addRequestHeader("Origin", URL);
    httpClient.executeMethod(getMethod);
    LOG.info("{} - {}", getMethod.getStatusCode(), getMethod.getStatusText());
    return getMethod;
  }

  protected static boolean checkIfServerIsRunning() {
    GetMethod request = null;
    boolean isRunning = false;
    try {
      request = httpGet("/");
      isRunning = request.getStatusCode() == 200;
    } catch (IOException e) {
      LOG.warn("AbstractTestRestApi.checkIfServerIsRunning() fails .. " +
          "Submarine server is not running");
      isRunning = false;
    } finally {
      if (request != null) {
        request.releaseConnection();
      }
    }
    return isRunning;
  }

  private static String getCookie(String user, String password) throws IOException {
    HttpClient httpClient = new HttpClient();
    PostMethod postMethod = new PostMethod(URL + "/login");
    postMethod.addRequestHeader("Origin", URL);
    postMethod.setParameter("password", password);
    postMethod.setParameter("userName", user);
    httpClient.executeMethod(postMethod);
    LOG.info("{} - {}", postMethod.getStatusCode(), postMethod.getStatusText());
    Pattern pattern = Pattern.compile("JSESSIONID=([a-zA-Z0-9-]*)");
    Header[] setCookieHeaders = postMethod.getResponseHeaders("Set-Cookie");
    String jsessionId = null;
    for (Header setCookie : setCookieHeaders) {
      java.util.regex.Matcher matcher = pattern.matcher(setCookie.toString());
      if (matcher.find()) {
        jsessionId = matcher.group(1);
      }
    }

    if (jsessionId != null) {
      return jsessionId;
    } else {
      return StringUtils.EMPTY;
    }
  }

  protected static boolean userAndPasswordAreNotBlank(String user, String pwd) {
    if (StringUtils.isBlank(user) && StringUtils.isBlank(pwd)) {
      return false;
    }
    return true;
  }
}
