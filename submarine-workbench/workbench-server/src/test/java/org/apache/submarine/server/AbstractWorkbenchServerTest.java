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

import com.google.gson.JsonElement;
import com.google.gson.JsonParseException;
import com.google.gson.JsonParser;
import org.apache.commons.httpclient.HttpClient;
import org.apache.commons.httpclient.methods.GetMethod;
import org.apache.commons.io.FileUtils;
import org.apache.submarine.commons.utils.SubmarineConfiguration;
import org.apache.submarine.utils.TestUtils;
import org.hamcrest.Description;
import org.hamcrest.TypeSafeMatcher;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.net.InetAddress;
import java.net.UnknownHostException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public abstract class AbstractWorkbenchServerTest {
  protected static final Logger LOG =
      LoggerFactory.getLogger(AbstractWorkbenchServerTest.class);

  static final String WEBSOCKET_API_URL = "/ws";
  static final String URL = getUrlToTest();
  protected static final boolean WAS_RUNNING = checkIfServerIsRunning();

  protected static File workbenchServerHome;
  protected static File confDir;

  public static String getWebsocketApiUrlToTest() {
    String websocketUrl = "ws://localhost:8080" + WEBSOCKET_API_URL;
    if (System.getProperty("websocketUrl") != null) {
      websocketUrl = System.getProperty("websocketurl");
    }
    return websocketUrl;
  }

  public static String getUrlToTest() {
    String url = "http://localhost:8080";
    if (System.getProperty("url") != null) {
      url = System.getProperty("url");
    }
    return url;
  }

  static ExecutorService executor;
  protected static final Runnable SERVER = new Runnable() {
    @Override
    public void run() {
      try {
        TestUtils.clearInstances();
        WorkbenchServer.main(new String[]{""});
      } catch (Exception e) {
        LOG.error("Exception in WebDriverManager while getWebDriver ", e);
        throw new RuntimeException(e);
      }
    }
  };

  public static void startUp(String testClassName) throws Exception {
    LOG.info("Starting WorkbenchServer testClassName: {}", testClassName);

    if (!WAS_RUNNING) {
      // copy the resources files to a temp folder
      workbenchServerHome = new File("..");
      LOG.info("SUBMARINE_WORKBENCH_SERVER_HOME: "
          + workbenchServerHome.getAbsolutePath());
      confDir = new File(workbenchServerHome, "conf_" + testClassName);
      confDir.mkdirs();

      System.setProperty(SubmarineConfiguration.ConfVars.WORKBENCH_WEB_WAR.getVarName(),
          new File("../workbench-web/dist").getAbsolutePath());
      System.setProperty(SubmarineConfiguration.ConfVars.SUBMARINE_CONF_DIR.getVarName(),
          confDir.getAbsolutePath());

      // some test profile does not build workbench-web.
      // to prevent submarine workbench server starting up fail,
      // create workbench-web/dist directory
      new File("../workbench-web/dist").mkdirs();

      LOG.info("Staring test workbench server up...");

      executor = Executors.newSingleThreadExecutor();
      executor.submit(SERVER);
      long s = System.currentTimeMillis();
      boolean started = false;
      while (System.currentTimeMillis() - s < 1000 * 60 * 3) {  // 3 minutes
        Thread.sleep(2000);
        started = checkIfServerIsRunning();
        if (started == true) {
          break;
        }
      }
      if (started == false) {
        throw new RuntimeException("Can not start workbench server.");
      }
      LOG.info("Test workbench server stared.");
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
      LOG.info("Terminating test workbench server...");
      WorkbenchServer.jettyWebServer.stop();
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
        throw new RuntimeException("Can not stop Submarine workbench server");
      }

      LOG.info("Test Submarine workbench server terminated.");

      if (deleteConfDir) {
        FileUtils.deleteDirectory(confDir);
      }
    }
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
          "Submarine workbench server is not running");
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
            .appendText("\" beeing a root element ");
      }

      @Override
      protected void describeMismatchSafely(JsonElement root, Description description) {
        description.appendText("got ").appendText(root.toString());
      }
    };
  }
}
