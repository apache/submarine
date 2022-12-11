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
package org.apache.submarine.server;

import org.apache.commons.lang3.StringUtils;
import org.apache.log4j.PropertyConfigurator;
import org.apache.submarine.server.database.utils.MyBatisUtil;
import org.apache.submarine.server.rest.provider.YamlEntityProvider;
import org.apache.submarine.server.security.SecurityFactory;
import org.apache.submarine.server.security.SecurityProvider;
import org.apache.submarine.server.security.common.AuthFlowType;
import org.apache.submarine.server.workbench.websocket.NotebookServer;
import org.apache.submarine.server.websocket.WebSocketServer;
import org.eclipse.jetty.http.HttpCookie;
import org.eclipse.jetty.http.HttpVersion;
import org.eclipse.jetty.server.Handler;
import org.eclipse.jetty.server.HttpConfiguration;
import org.eclipse.jetty.server.HttpConnectionFactory;
import org.eclipse.jetty.server.SecureRequestCustomizer;
import org.eclipse.jetty.server.Server;
import org.eclipse.jetty.server.ServerConnector;
import org.eclipse.jetty.server.SslConnectionFactory;
import org.eclipse.jetty.server.handler.HandlerList;
import org.eclipse.jetty.server.session.DatabaseAdaptor;
import org.eclipse.jetty.server.session.JDBCSessionDataStoreFactory;
import org.eclipse.jetty.server.session.SessionHandler;
import org.eclipse.jetty.servlet.DefaultServlet;
import org.eclipse.jetty.servlet.ServletContextHandler;
import org.eclipse.jetty.servlet.ServletHolder;
import org.eclipse.jetty.util.ssl.SslContextFactory;
import org.eclipse.jetty.util.thread.QueuedThreadPool;
import org.eclipse.jetty.util.thread.ThreadPool;
import org.eclipse.jetty.webapp.WebAppContext;
import org.eclipse.jetty.websocket.servlet.WebSocketServlet;
import org.glassfish.hk2.api.ServiceLocator;
import org.glassfish.hk2.api.ServiceLocatorFactory;
import org.glassfish.hk2.utilities.ServiceLocatorUtilities;
import org.glassfish.hk2.utilities.binding.AbstractBinder;
import org.glassfish.jersey.server.ResourceConfig;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import org.apache.submarine.commons.utils.SubmarineConfiguration;
import org.apache.submarine.commons.utils.SubmarineConfVars;

import javax.inject.Inject;
import javax.inject.Singleton;
import javax.servlet.DispatcherType;
import javax.servlet.Filter;
import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.EnumSet;
import java.util.Optional;

public class SubmarineServer extends ResourceConfig {
  private static final Logger LOG = LoggerFactory.getLogger(SubmarineServer.class);

  private static final long SERVERTIMESTAMP = System.currentTimeMillis();

  public static Server jettyWebServer;
  public static ServiceLocator sharedServiceLocator;
  private static WebAppContext webApp;
  private static final SubmarineConfiguration conf = SubmarineConfiguration.getInstance();

  public static long getServerTimeStamp() {
    return SERVERTIMESTAMP;
  }

  public static void main(String[] args) throws InterruptedException {
    PropertyConfigurator.configure(ClassLoader.getSystemResource("log4j.properties"));

    LOG.info("Submarine server Host: " + conf.getServerAddress());
    if (!conf.useSsl()) {
      LOG.info("Submarine server Port: " + conf.getServerPort());
    } else {
      LOG.info("Submarine server SSL Port: " + conf.getServerSslPort());
    }

    jettyWebServer = setupJettyServer(conf);

    // Web UI
    HandlerList handlers = new HandlerList();
    webApp = setupWebAppContext(handlers, conf);
    jettyWebServer.setHandler(handlers);

    // Add
    sharedServiceLocator = ServiceLocatorFactory.getInstance().create("shared-locator");
    ServiceLocatorUtilities.enableImmediateScope(sharedServiceLocator);
    ServiceLocatorUtilities.bind(
        sharedServiceLocator,
        new AbstractBinder() {
          @Override
          protected void configure() {
            bindAsContract(NotebookServer.class)
                .to(WebSocketServlet.class)
                .in(Singleton.class);
            bindAsContract(WebSocketServer.class)
                .to(WebSocketServlet.class)
                .in(Singleton.class);
          }
        });

    setupRestApiContextHandler(webApp, conf);

    // Cookie config
    setCookieConfig(webApp);

    // Notebook server
    setupNotebookServer(webApp, conf, sharedServiceLocator);

    // Cluster Server
    // Cluster Server is useless for submarine now. Shield it to improve performance.
    // setupClusterServer();

    setupWebSocketServer(webApp, conf, sharedServiceLocator);
    startServer();

  }

  @Inject
  public SubmarineServer() {
    packages("org.apache.submarine.server.workbench.rest",
        "org.apache.submarine.server.rest"
    );
    register(YamlEntityProvider.class);
  }

  private static void startServer() throws InterruptedException {
    LOG.info("Starting submarine server");
    try {
      // Instantiates SubmarineServer
      jettyWebServer.start();
    } catch (Exception e) {
      LOG.error("Error while running jettyServer", e);
      System.exit(-1);
    }
    LOG.info("Done, submarine server started");

    Runtime.getRuntime()
        .addShutdownHook(
            new Thread(
                () -> {
                  LOG.info("Shutting down Submarine Server ... ");
                  try {
                    jettyWebServer.stop();
                    Thread.sleep(3000);
                  } catch (InterruptedException e) {
                    LOG.error("Interrupted exception:", e);
                  } catch (Exception e) {
                    LOG.error("Error while stopping servlet container", e);
                  }
                  LOG.info("Bye");
                }));

    jettyWebServer.join();
  }

  private static void setupRestApiContextHandler(WebAppContext webapp, SubmarineConfiguration conf) {
    final ServletHolder servletHolder =
        new ServletHolder(new org.glassfish.jersey.servlet.ServletContainer());

    servletHolder.setInitParameter("javax.ws.rs.Application", SubmarineServer.class.getName());
    servletHolder.setName("rest");
    servletHolder.setForcedPath("rest");
    webapp.setSessionHandler(new SessionHandler());
    webapp.addServlet(servletHolder, "/api/*");
  }

  private static WebAppContext setupWebAppContext(HandlerList handlers,
                                                  SubmarineConfiguration conf) {
    WebAppContext webApp = new WebAppContext();
    webApp.setContextPath("/");
    File warPath = new File(conf.getString(SubmarineConfVars.ConfVars.WORKBENCH_WEB_WAR));
    LOG.info("workbench web war file path is {}.",
        conf.getString(SubmarineConfVars.ConfVars.WORKBENCH_WEB_WAR));
    if (warPath.isDirectory()) {
      // Development mode, read from FS
      webApp.setResourceBase(warPath.getPath());
      webApp.setParentLoaderPriority(true);
    } else {
      // use packaged WAR
      webApp.setWar(warPath.getAbsolutePath());
      File warTempDirectory = new File("webapps");
      warTempDirectory.mkdir();
      webApp.setTempDirectory(warTempDirectory);
    }

    // add security filter
    Optional<SecurityProvider> securityProvider = SecurityFactory.getSecurityProvider();
    if (securityProvider.isPresent()) {
      SecurityProvider provider = securityProvider.get();
      Class<Filter> filterClass = provider.getFilterClass();
      // add filter
      LOG.info("Add {} to support auth", filterClass);
      webApp.addFilter(filterClass, "/*", EnumSet.of(DispatcherType.REQUEST));
      // add flow type result to front end
      AuthFlowType type = provider.getAuthFlowType();
      // If using session, we can add JDBCSessionDataStoreFactory to support clustering session
      // This solves two problems:
      // 1. session loss after service restart
      // 2. session sharing when multiple replicas
      if (type == AuthFlowType.SESSION) {
        // Configure a JDBCSessionDataStoreFactory.
        JDBCSessionDataStoreFactory sessionDataStoreFactory = new JDBCSessionDataStoreFactory();
        sessionDataStoreFactory.setGracePeriodSec(3600);
        sessionDataStoreFactory.setSavePeriodSec(0);
        // add datasource (current mybatis) to factory
        DatabaseAdaptor adaptor = new DatabaseAdaptor();
        adaptor.setDatasource(MyBatisUtil.getDatasource());
        sessionDataStoreFactory.setDatabaseAdaptor(adaptor);
        // Add the SessionDataStoreFactory as a bean on the server.
        jettyWebServer.addBean(sessionDataStoreFactory);
      }

      ServletHolder authProviderServlet = new ServletHolder(new HttpServlet() {
        private static final long serialVersionUID = 1L;
        private final String staticProviderJs = String.format(
            "(function () { window.GLOBAL_CONFIG = { \"type\": \"%s\" }; })();", type.getType()
        );
        private static final String contentType = "application/javascript";
        private static final String encoding = "UTF-8";
        @Override
        protected void doGet(HttpServletRequest req, HttpServletResponse resp)
                throws ServletException, IOException {
          resp.setContentType(contentType);
          resp.setCharacterEncoding(encoding);
          resp.getWriter().write(staticProviderJs);
        }
      });
      webApp.addServlet(authProviderServlet, "/assets/security/provider.js");
    }

    webApp.addServlet(new ServletHolder(new DefaultServlet()), "/");
    // When requesting the workbench page, the content of index.html needs to be returned,
    // otherwise a 404 error will be displayed
    // NOTE: If you modify the workbench directory in the front-end URL,
    // you need to modify the `/workbench/*` here.
    webApp.addServlet(new ServletHolder(RefreshServlet.class), "/user/*");
    webApp.addServlet(new ServletHolder(RefreshServlet.class), "/workbench/*");

    handlers.setHandlers(new Handler[]{webApp});

    return webApp;
  }

  /**
   * Session cookie config
   */
  public static void setCookieConfig(WebAppContext webapp) {
    // http only
    webapp.getSessionHandler().getSessionCookieConfig().setHttpOnly(
        conf.getBoolean(SubmarineConfVars.ConfVars.SUBMARINE_COOKIE_HTTP_ONLY)
    );
    // same site: NONE("None"), STRICT("Strict"), LAX("Lax");
    String sameSite = conf.getString(SubmarineConfVars.ConfVars.SUBMARINE_COOKIE_SAMESITE);
    if (StringUtils.isNoneBlank(sameSite)) {
      webapp.getSessionHandler().setSameSite(HttpCookie.SameSite.valueOf(sameSite.toUpperCase()));
    }
    // secure
    webapp.getSessionHandler().getSessionCookieConfig().setSecure(
        conf.getBoolean(SubmarineConfVars.ConfVars.SUBMARINE_COOKIE_SECURE)
    );
  }

  private static Server setupJettyServer(SubmarineConfiguration conf) {
    ThreadPool threadPool =
        new QueuedThreadPool(conf.getInt(SubmarineConfVars.ConfVars.SUBMARINE_SERVER_JETTY_THREAD_POOL_MAX),
            conf.getInt(SubmarineConfVars.ConfVars.SUBMARINE_SERVER_JETTY_THREAD_POOL_MIN),
            conf.getInt(SubmarineConfVars.ConfVars.SUBMARINE_SERVER_JETTY_THREAD_POOL_TIMEOUT));
    final Server server = new Server(threadPool);
    ServerConnector connector;

    if (conf.useSsl()) {
      LOG.debug("Enabling SSL for submarine Server on port " + conf.getServerSslPort());
      HttpConfiguration httpConfig = new HttpConfiguration();
      httpConfig.setSecureScheme("https");
      httpConfig.setSecurePort(conf.getServerSslPort());
      httpConfig.setOutputBufferSize(32768);
      httpConfig.setResponseHeaderSize(8192);
      httpConfig.setSendServerVersion(true);

      HttpConfiguration httpsConfig = new HttpConfiguration(httpConfig);
      SecureRequestCustomizer src = new SecureRequestCustomizer();
      httpsConfig.addCustomizer(src);

      connector = new ServerConnector(
          server,
          new SslConnectionFactory(getSslContextFactory(conf), HttpVersion.HTTP_1_1.asString()),
          new HttpConnectionFactory(httpsConfig));
    } else {
      connector = new ServerConnector(server);
    }

    configureRequestHeaderSize(conf, connector);
    // Set some timeout options to make debugging easier.
    int timeout = 1000 * 30;
    connector.setIdleTimeout(timeout);
    connector.setHost(conf.getServerAddress());
    if (conf.useSsl()) {
      connector.setPort(conf.getServerSslPort());
    } else {
      connector.setPort(conf.getServerPort());
    }

    server.addConnector(connector);
    return server;
  }

  private static void setupNotebookServer(WebAppContext webapp,
                                          SubmarineConfiguration conf, ServiceLocator serviceLocator) {
    String maxTextMessageSize = conf.getWebsocketMaxTextMessageSize();
    final ServletHolder servletHolder =
        new ServletHolder(serviceLocator.getService(NotebookServer.class));
    servletHolder.setInitParameter("maxTextMessageSize", maxTextMessageSize);

    final ServletContextHandler context = new ServletContextHandler(ServletContextHandler.SESSIONS);
    webapp.addServlet(servletHolder, "/wss/*");
  }

  private static void setupWebSocketServer(WebAppContext webapp,
                                           SubmarineConfiguration conf, ServiceLocator serviceLocator) {
    String maxTextMessageSize = conf.getWebsocketMaxTextMessageSize();
    final ServletHolder notebookServletHolder =
        new ServletHolder(serviceLocator.getService(WebSocketServer.class));
    notebookServletHolder.setInitParameter("maxTextMessageSize", maxTextMessageSize);

    final ServletHolder experimentServletHolder =
        new ServletHolder(serviceLocator.getService(WebSocketServer.class));
    experimentServletHolder.setInitParameter("maxTextMessageSize", maxTextMessageSize);

    final ServletHolder environmentServletHolder =
        new ServletHolder(serviceLocator.getService(WebSocketServer.class));
    environmentServletHolder.setInitParameter("maxTextMessageSize", maxTextMessageSize);



    final ServletContextHandler context = new ServletContextHandler(ServletContextHandler.SESSIONS);
    webapp.addServlet(notebookServletHolder, "/ws/notebook/*");
    webapp.addServlet(experimentServletHolder, "/ws/experiment/*");
    webapp.addServlet(environmentServletHolder, "/ws/environment/*");
  }

  private static SslContextFactory getSslContextFactory(SubmarineConfiguration conf) {
    SslContextFactory sslContextFactory = new SslContextFactory();

    // Set keystore
    sslContextFactory.setKeyStorePath(conf.getKeyStorePath());
    sslContextFactory.setKeyStoreType(conf.getKeyStoreType());
    sslContextFactory.setKeyStorePassword(conf.getKeyStorePassword());
    sslContextFactory.setKeyManagerPassword(conf.getKeyManagerPassword());

    if (conf.useClientAuth()) {
      sslContextFactory.setNeedClientAuth(conf.useClientAuth());

      // Set truststore
      sslContextFactory.setTrustStorePath(conf.getTrustStorePath());
      sslContextFactory.setTrustStoreType(conf.getTrustStoreType());
      sslContextFactory.setTrustStorePassword(conf.getTrustStorePassword());
    }

    return sslContextFactory;
  }

  private static void configureRequestHeaderSize(
      SubmarineConfiguration conf, ServerConnector connector) {
    HttpConnectionFactory cf =
        (HttpConnectionFactory) connector.getConnectionFactory(HttpVersion.HTTP_1_1.toString());
    int requestHeaderSize = conf.getJettyRequestHeaderSize();
    cf.getHttpConfiguration().setRequestHeaderSize(requestHeaderSize);
  }

  // SUBMARINE-422. Fix refreshing page returns 404 error
  // Because the workbench is developed using angular,
  // the adjustment of angular WEB pages is completely controlled by the front end,
  // so when you manually refresh a specific page in the browser,
  // the browser will send the request for this page to the back-end service,
  // but the back-end service only In response to API requests, it will cause the front end to display 404.
  // The solution is to find that not all API requests directly return the content of the index page,
  // so that the front end will automatically perform correct page routing processing.
  public static class RefreshServlet extends HttpServlet {
    private static final long serialVersionUID = 1L;

    @Override
    protected void doGet(HttpServletRequest request, HttpServletResponse response)
        throws ServletException, IOException {
      response.setContentType("text/html");
      response.encodeRedirectURL("/");
      response.setStatus(HttpServletResponse.SC_OK);

      File warPath = new File(conf.getString(SubmarineConfVars.ConfVars.WORKBENCH_WEB_WAR));
      File indexFile = null;
      if (warPath.isDirectory()) {
        // Development mode, read from FS
        indexFile = new File(warPath.getAbsolutePath() + "/index.html");
      } else {
        // Product mode, read from war file
        File warFile = webApp.getTempDirectory();
        if (!warFile.exists()) {
          throw new ServletException("Can't found war directory!");
        }
        indexFile = new File(warFile.getAbsolutePath() + "/webapp/index.html");
      }

      // If index.html does not exist, throw ServletException
      if (!(indexFile.isFile() && indexFile.exists())) {
        throw new ServletException("Can't found index html!");
      }

      StringBuilder sbIndexBuf = new StringBuilder();
      try (InputStreamReader reader =
               new InputStreamReader(new FileInputStream(indexFile), "GBK");
           BufferedReader bufferedReader = new BufferedReader(reader);) {
        String lineTxt = null;
        while ((lineTxt = bufferedReader.readLine()) != null) {
          sbIndexBuf.append(lineTxt);
        }
      } catch (IOException e) {
        LOG.error(e.getMessage(), e);
      }

      response.getWriter().print(sbIndexBuf.toString());
    }
  }
}
