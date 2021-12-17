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
package org.apache.submarine.server;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.InetAddress;
import java.net.UnknownHostException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.regex.Pattern;

import javax.ws.rs.core.MediaType;
import javax.ws.rs.core.Response;

import org.apache.commons.httpclient.Header;
import org.apache.commons.httpclient.HttpClient;
import org.apache.commons.httpclient.cookie.CookiePolicy;
import org.apache.commons.httpclient.methods.ByteArrayRequestEntity;
import org.apache.commons.httpclient.methods.DeleteMethod;
import org.apache.commons.httpclient.methods.GetMethod;
import org.apache.commons.httpclient.methods.PostMethod;
import org.apache.commons.httpclient.methods.PutMethod;
import org.apache.commons.httpclient.methods.RequestEntity;
import org.apache.commons.io.FileUtils;
import org.apache.commons.lang3.StringUtils;
import org.apache.http.HttpEntity;
import org.apache.http.HttpResponse;
import org.apache.http.client.ClientProtocolException;
import org.apache.http.client.ResponseHandler;
import org.apache.http.client.methods.HttpPatch;
import org.apache.http.entity.ContentType;
import org.apache.http.entity.StringEntity;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClients;
import org.apache.http.util.EntityUtils;
import org.apache.submarine.commons.utils.SubmarineConfVars;
import org.apache.submarine.server.api.environment.Environment;
import org.apache.submarine.server.api.environment.EnvironmentId;
import org.apache.submarine.server.gson.EnvironmentIdDeserializer;
import org.apache.submarine.server.gson.EnvironmentIdSerializer;
import org.apache.submarine.server.response.JsonResponse;
import org.apache.submarine.server.rest.RestConstants;
import org.apache.submarine.server.security.SecurityFactory;
import org.apache.submarine.server.security.test.TestSecurityProvider;
import org.apache.submarine.server.utils.TestUtils;
import org.hamcrest.Description;
import org.hamcrest.TypeSafeMatcher;
import org.junit.Assert;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonElement;
import com.google.gson.JsonParseException;
import com.google.gson.JsonParser;

public abstract class AbstractSubmarineServerTest {
  protected static final Logger LOG =
      LoggerFactory.getLogger(AbstractSubmarineServerTest.class);

  static final String WEBSOCKET_API_URL = "/ws";
  static final String URL = getUrlToTest();
  protected static final boolean WAS_RUNNING = checkIfServerIsRunning();

  protected static File submarineServerHome;
  protected static File confDir;

  protected static String ENV_PATH =
      "/api/" + RestConstants.V1 + "/" + RestConstants.ENVIRONMENT;
  protected static String ENV_NAME = "my-submarine-env";

  public static String getWebsocketApiUrlToTest() {
    String websocketUrl = "ws://localhost:8080" + WEBSOCKET_API_URL;
    if (System.getProperty("websocketUrl") != null) {
      websocketUrl = System.getProperty("websocketurl");
    }
    LOG.debug("getWebsocketApiUrlToTest = {}", websocketUrl);
    return websocketUrl;
  }

  public static String getUrlToTest() {
    String url = "http://localhost:8080";
    if (System.getProperty("url") != null) {
      url = System.getProperty("url");
    }
    LOG.debug("getUrlToTest = {}", url);
    return url;
  }

  static ExecutorService executor;
  protected static final Runnable SERVER = new Runnable() {
    @Override
    public void run() {
      try {
        TestUtils.clearInstances();
        SubmarineServer.main(new String[]{""});
      } catch (Exception e) {
        LOG.error("Exception in WebDriverManager while getWebDriver ", e);
        throw new RuntimeException(e);
      }
    }
  };

  public static void startUp(String testClassName) throws Exception {
    LOG.info("Starting SubmarineServer testClassName: {}", testClassName);

    if (!WAS_RUNNING) {
      // add test filter
      System.setProperty(SubmarineConfVars.ConfVars.SUBMARINE_AUTH_TYPE.getVarName(), "test");
      SecurityFactory.addProvider("test", new TestSecurityProvider());

      // copy the resources files to a temp folder
      submarineServerHome = new File("..");
      LOG.info("SUBMARINE_SERVER_HOME: "
          + submarineServerHome.getAbsolutePath());
      confDir = new File(submarineServerHome, "conf_" + testClassName);
      confDir.mkdirs();

      System.setProperty(SubmarineConfVars.ConfVars.WORKBENCH_WEB_WAR.getVarName(),
          new File("../../submarine-workbench/workbench-web/dist/workbench-web").getAbsolutePath());
      System.setProperty(SubmarineConfVars.ConfVars.SUBMARINE_CONF_DIR.getVarName(),
          confDir.getAbsolutePath());

      // some test profile does not build workbench-web.
      // to prevent submarine server starting up fail,
      // create workbench-web/dist directory
      new File("../../submarine-workbench/workbench-web/dist/workbench-web").mkdirs();

      LOG.info("Staring test Submarine server up...");

      executor = Executors.newSingleThreadExecutor();
      executor.submit(SERVER);
      long s = System.currentTimeMillis();
      boolean started = false;
      while (System.currentTimeMillis() - s < 1000 * 60 * 5) {  // 5 minutes
        Thread.sleep(2000);
        started = checkIfServerIsRunning();
        if (started == true) {
          break;
        }
      }
      if (started == false) {
        throw new RuntimeException("Can not start Submarine server.");
      }
      LOG.info("Test Submarine server stared.");
    }
  }

  private static String getHostname() {
    try {
      return InetAddress.getLocalHost().getHostName();
    } catch (UnknownHostException e) {
      LOG.error("Exception in WebDriverManager while getWebDriver ", e);
      return "localhost";
    }
  }

  public static void shutDown() throws Exception {
    shutDown(true);
  }

  protected static void shutDown(final boolean deleteConfDir) throws Exception {
    if (!WAS_RUNNING) {
      LOG.info("Terminating test Submarine server...");
      SubmarineServer.jettyWebServer.stop();
      executor.shutdown();

      long s = System.currentTimeMillis();
      boolean started = true;
      // 3 minutes
      while (System.currentTimeMillis() - s < 1000 * 60 * 3) {
        Thread.sleep(2000);
        started = checkIfServerIsRunning();
        if (started == false) {
          break;
        }
      }
      if (started == true) {
        throw new RuntimeException("Can not stop Submarine server");
      }

      LOG.info("Test Submarine server terminated.");

      if (deleteConfDir) {
        FileUtils.deleteDirectory(confDir);
      }
    }
  }

  protected static PostMethod httpPost(String path, String body) throws IOException {
    return httpPost(path, body, StringUtils.EMPTY, StringUtils.EMPTY);
  }

  protected static PostMethod httpPost(String path, String body, String mediaType)
      throws IOException {
    return httpPost(path, body, mediaType, StringUtils.EMPTY, StringUtils.EMPTY);
  }

  protected static PostMethod httpPost(String path, String request, String user, String pwd)
      throws IOException {
    return httpPost(path, request, MediaType.APPLICATION_JSON, user, pwd);
  }

  protected static PostMethod httpPost(String path, String request, String mediaType, String user,
                                       String pwd) throws IOException {
    LOG.info("Connecting to {}", URL + path);

    HttpClient httpClient = new HttpClient();
    PostMethod postMethod = new PostMethod(URL + path);
    postMethod.setRequestBody(request);
    postMethod.setRequestHeader("Content-type", mediaType);
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

  protected static String httpPatch(String path, String body,
                                    String contentType) throws IOException {
    LOG.info("Connecting to {}", URL + path);
    CloseableHttpClient httpclient = HttpClients.createDefault();

    HttpPatch httpPatch = new HttpPatch(URL + path);
    StringEntity entity = new StringEntity(body, ContentType.APPLICATION_JSON);
    httpPatch.setEntity(entity);

    // Create a custom response handler
    ResponseHandler<String> responseHandler = new ResponseHandler<String>() {
      @Override
      public String handleResponse(final HttpResponse response)
          throws ClientProtocolException, IOException {
        LOG.info("response " + response);
        int status = response.getStatusLine().getStatusCode();
        if (status >= 200 && status < 300) {
          HttpEntity entity = response.getEntity();
          return entity != null ? EntityUtils.toString(entity) : null;
        } else {
          throw new ClientProtocolException(
              "Unexpected response status: " + status);
        }
      }
    };
    String responseBody = httpclient.execute(httpPatch, responseHandler);
    httpclient.close();
    return responseBody;
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

  protected TypeSafeMatcher<String> isJSON() {
    return new TypeSafeMatcher<String>() {
      @Override
      public boolean matchesSafely(String body) {
        String b = body.trim();
        return (b.startsWith("{") && b.endsWith("}")) || (b.startsWith("[") && b.endsWith("]"));
      }

      @Override
      public void describeTo(Description description) {
        description.appendText("response in JSON format ");
      }

      @Override
      protected void describeMismatchSafely(String item, Description description) {
        description.appendText("got ").appendText(item);
      }
    };
  }

  protected TypeSafeMatcher<String> isValidJSON() {
    return new TypeSafeMatcher<String>() {
      @Override
      public boolean matchesSafely(String body) {
        boolean isValid = true;
        try {
          new JsonParser().parse(body);
        } catch (JsonParseException e) {
          LOG.error("Exception in AbstractTestRestApi while matchesSafely ", e);
          isValid = false;
        }
        return isValid;
      }

      @Override
      public void describeTo(Description description) {
        description.appendText("response in JSON format ");
      }

      @Override
      protected void describeMismatchSafely(String item, Description description) {
        description.appendText("got ").appendText(item);
      }
    };
  }

  protected TypeSafeMatcher<? super JsonElement> hasRootElementNamed(final String memberName) {
    return new TypeSafeMatcher<JsonElement>() {
      @Override
      protected boolean matchesSafely(JsonElement item) {
        return item.isJsonObject() && item.getAsJsonObject().has(memberName);
      }

      @Override
      public void describeTo(Description description) {
        description.appendText("response in JSON format with \"").appendText(memberName)
            .appendText("\" being a root element ");
      }

      @Override
      protected void describeMismatchSafely(JsonElement root, Description description) {
        description.appendText("got ").appendText(root.toString());
      }
    };
  }

  protected static boolean userAndPasswordAreNotBlank(String user, String pwd) {
    if (StringUtils.isBlank(user) && StringUtils.isBlank(pwd)) {
      return false;
    }
    return true;
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

  protected void run(String body, String contentType) throws Exception {
    Gson gson = new GsonBuilder()
        .registerTypeAdapter(EnvironmentId.class, new EnvironmentIdSerializer())
        .registerTypeAdapter(EnvironmentId.class, new EnvironmentIdDeserializer())
        .create();

    // create
    LOG.info("Create Environment using Environment REST API");

    PostMethod postMethod = httpPost(ENV_PATH, body, contentType);
    Assert.assertEquals(Response.Status.OK.getStatusCode(),
        postMethod.getStatusCode());

    String json = postMethod.getResponseBodyAsString();
    JsonResponse jsonResponse = gson.fromJson(json, JsonResponse.class);
    Assert.assertEquals(Response.Status.OK.getStatusCode(),
        jsonResponse.getCode());

    Environment env =
        gson.fromJson(gson.toJson(jsonResponse.getResult()), Environment.class);
    verifyCreateEnvironmentApiResult(env);
  }

  protected void update(String body, String contentType) throws Exception {
    Gson gson = new GsonBuilder()
        .registerTypeAdapter(EnvironmentId.class, new EnvironmentIdSerializer())
        .registerTypeAdapter(EnvironmentId.class, new EnvironmentIdDeserializer())
        .create();

    // update
    LOG.info("Update Environment using Environment REST API");

    String json = httpPatch(ENV_PATH + "/" + ENV_NAME, body, contentType);

    JsonResponse jsonResponse = gson.fromJson(json, JsonResponse.class);
    Assert.assertEquals(Response.Status.OK.getStatusCode(),
        jsonResponse.getCode());

    Environment env =
        gson.fromJson(gson.toJson(jsonResponse.getResult()), Environment.class);
    verifyUpdateEnvironmentApiResult(env);
  }

  protected void verifyCreateEnvironmentApiResult(Environment env)
      throws Exception {
    Assert.assertNotNull(env.getEnvironmentSpec().getName());
    Assert.assertNotNull(env.getEnvironmentSpec());
  }

  protected void verifyUpdateEnvironmentApiResult(Environment env)
      throws Exception {
    Assert.assertEquals(env.getEnvironmentSpec().getDockerImage(),
        "continuumio/miniconda3");
    Assert.assertTrue(
        env.getEnvironmentSpec().getKernelSpec().getCondaDependencies().size() == 1);
  }


  protected void deleteEnvironment() throws IOException {
    Gson gson = new GsonBuilder()
        .registerTypeAdapter(EnvironmentId.class, new EnvironmentIdSerializer())
        .registerTypeAdapter(EnvironmentId.class, new EnvironmentIdDeserializer())
        .create();
    DeleteMethod deleteMethod = httpDelete(ENV_PATH + "/" + ENV_NAME);
    Assert.assertEquals(Response.Status.OK.getStatusCode(),
        deleteMethod.getStatusCode());

    String json = deleteMethod.getResponseBodyAsString();
    JsonResponse jsonResponse = gson.fromJson(json, JsonResponse.class);
    Assert.assertEquals(Response.Status.OK.getStatusCode(),
        jsonResponse.getCode());

    Environment deletedEnv =
        gson.fromJson(gson.toJson(jsonResponse.getResult()), Environment.class);
    Assert.assertEquals(ENV_NAME, deletedEnv.getEnvironmentSpec().getName());
  }

  protected String loadContent(String resourceName) throws Exception {
    StringBuilder content = new StringBuilder();
    InputStream inputStream =
        this.getClass().getClassLoader().getResourceAsStream(resourceName);
    BufferedReader r = new BufferedReader(new InputStreamReader(inputStream));
    String l;
    while ((l = r.readLine()) != null) {
      content.append(l).append("\n");
    }
    inputStream.close();
    return content.toString();
  }
}
