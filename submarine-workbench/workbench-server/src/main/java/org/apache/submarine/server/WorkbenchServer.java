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

import org.apache.log4j.PropertyConfigurator;
import org.apache.submarine.websocket.NotebookServer;
import org.apache.submarine.commons.cluster.ClusterServer;
import org.eclipse.jetty.http.HttpVersion;
import org.eclipse.jetty.server.HttpConfiguration;
import org.eclipse.jetty.server.HttpConnectionFactory;
import org.eclipse.jetty.server.SecureRequestCustomizer;
import org.eclipse.jetty.server.Server;
import org.eclipse.jetty.server.ServerConnector;
import org.eclipse.jetty.server.SslConnectionFactory;
import org.eclipse.jetty.server.handler.ContextHandlerCollection;
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
import org.apache.submarine.commons.utils.SubmarineConfiguration.ConfVars;

import javax.inject.Inject;
import javax.inject.Singleton;
import java.io.File;

public class WorkbenchServer extends ResourceConfig {
  private static final Logger LOG = LoggerFactory.getLogger(WorkbenchServer.class);

  public static Server jettyWebServer;
  public static ServiceLocator sharedServiceLocator;

  private static SubmarineConfiguration conf = SubmarineConfiguration.getInstance();

  public static void main(String[] args) throws InterruptedException {
    PropertyConfigurator.configure(ClassLoader.getSystemResource("log4j.properties"));

    final SubmarineConfiguration conf = SubmarineConfiguration.getInstance();
    LOG.info("Workbench server Host: " + conf.getServerAddress());
    if (conf.useSsl() == false) {
      LOG.info("Workbench server Port: " + conf.getServerPort());
    } else {
      LOG.info("Workbench server SSL Port: " + conf.getServerSslPort());
    }

    jettyWebServer = setupJettyServer(conf);

    ContextHandlerCollection contexts = new ContextHandlerCollection();
    jettyWebServer.setHandler(contexts);

    // Web UI
    final WebAppContext webApp = setupWebAppContext(contexts, conf);

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
          }
        });

    setupRestApiContextHandler(webApp, conf);
    // Notebook server
    setupNotebookServer(webApp, conf, sharedServiceLocator);

    // Cluster Server
    setupClusterServer();

    startServer();
  }

  @Inject
  public WorkbenchServer() {
    packages("org.apache.submarine.rest");
  }

  private static void startServer() throws InterruptedException {
    LOG.info("Starting submarine server");
    try {
      jettyWebServer.start(); // Instantiates WorkbenchServer
    } catch (Exception e) {
      LOG.error("Error while running jettyServer", e);
      System.exit(-1);
    }
    LOG.info("Done, submarine server started");

    Runtime.getRuntime()
        .addShutdownHook(
            new Thread(
                () -> {
                  LOG.info("Shutting down Submarine Workbench Server ... ");
                  try {
                    jettyWebServer.stop();
                    Thread.sleep(3000);
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

    servletHolder.setInitParameter("javax.ws.rs.Application", WorkbenchServer.class.getName());
    servletHolder.setName("rest");
    servletHolder.setForcedPath("rest");
    webapp.setSessionHandler(new SessionHandler());
    webapp.addServlet(servletHolder, "/api/*");
  }

  private static WebAppContext setupWebAppContext(ContextHandlerCollection contexts,
      SubmarineConfiguration conf) {
    WebAppContext webApp = new WebAppContext();
    webApp.setContextPath("/");
    File warPath = new File(conf.getString(ConfVars.WORKBENCH_WEB_WAR));
    LOG.info("workbench web war file path is {}.", conf.getString(ConfVars.WORKBENCH_WEB_WAR));
    if (warPath.isDirectory()) {
      // Development mode, read from FS
      // webApp.setDescriptor(warPath+"/WEB-INF/web.xml");
      webApp.setResourceBase(warPath.getPath());
      webApp.setParentLoaderPriority(true);
    } else {
      // use packaged WAR
      webApp.setWar(warPath.getAbsolutePath());
      File warTempDirectory = new File("webapps");
      warTempDirectory.mkdir();
      webApp.setTempDirectory(warTempDirectory);
    }
    // Explicit bind to root
    webApp.addServlet(new ServletHolder(new DefaultServlet()), "/*");
    contexts.addHandler(webApp);

    return webApp;
  }

  private static Server setupJettyServer(SubmarineConfiguration conf) {
    ThreadPool threadPool =
        new QueuedThreadPool(conf.getInt(ConfVars.SERVER_JETTY_THREAD_POOL_MAX),
            conf.getInt(ConfVars.SERVER_JETTY_THREAD_POOL_MIN),
            conf.getInt(ConfVars.SERVER_JETTY_THREAD_POOL_TIMEOUT));
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
    connector.setSoLingerTime(-1);
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
    webapp.addServlet(servletHolder, "/ws/*");
  }

  private static void setupClusterServer() {
    if (conf.workbenchIsClusterMode()) {
      ClusterServer clusterServer = ClusterServer.getInstance();
      clusterServer.start();
    }
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

}
