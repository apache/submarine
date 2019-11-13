/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 * <p>
 * http://www.apache.org/licenses/LICENSE-2.0
 * <p>
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.submarine.jobserver;


import org.eclipse.jetty.http.HttpVersion;
import org.eclipse.jetty.server.HttpConfiguration;
import org.eclipse.jetty.server.HttpConnectionFactory;
import org.eclipse.jetty.server.SecureRequestCustomizer;
import org.eclipse.jetty.server.Server;
import org.eclipse.jetty.server.ServerConnector;
import org.eclipse.jetty.server.SslConnectionFactory;
import org.eclipse.jetty.servlet.ServletContextHandler;
import org.eclipse.jetty.servlet.ServletHolder;
import org.eclipse.jetty.util.ssl.SslContextFactory;
import org.eclipse.jetty.util.thread.QueuedThreadPool;
import org.eclipse.jetty.util.thread.ThreadPool;
import org.glassfish.jersey.servlet.ServletContainer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.apache.submarine.commons.utils.SubmarineConfiguration;


/**
 * The ml job server. It will load the classes in rest package when
 * bootstrap with related configurable settings.
 * */
public class JobServer {

  private static final Logger LOG = LoggerFactory.getLogger(JobServer.class);

  private SubmarineConfiguration conf = SubmarineConfiguration.getInstance();

  private Server jobServer;

  public void start() {
    ServletContextHandler context = new
        ServletContextHandler(ServletContextHandler.SESSIONS);
    context.setContextPath("/");
    setupServer(conf);;
    jobServer.setHandler(context);

    // Job API servlet
    ServletHolder apiServlet = context.addServlet(ServletContainer.class,
        conf.getJobServerUrlPrefix());
    apiServlet.setInitOrder(1);
    apiServlet.setInitParameter("jersey.config.server.provider.packages",
        "org.apache.submarine.jobserver.rest");

    try {
      jobServer.start();
      LOG.info("Submarine job server started");
      jobServer.join();
    } catch (Exception e) {
      LOG.error("Submarine job server failed to start");
      e.printStackTrace();
    } finally {
      jobServer.destroy();
      LOG.info("Submarine job server stopped");
    }
  }

  public static void main(String[] args) throws Exception {
    new JobServer().start();
  }

  private Server setupServer(SubmarineConfiguration conf) {
    ThreadPool threadPool =
        new QueuedThreadPool(
            conf.getInt(SubmarineConfiguration
                .ConfVars.SERVER_JETTY_THREAD_POOL_MAX),
            conf.getInt(SubmarineConfiguration
                .ConfVars.SERVER_JETTY_THREAD_POOL_MIN),
            conf.getInt(SubmarineConfiguration
                .ConfVars.SERVER_JETTY_THREAD_POOL_TIMEOUT));
    jobServer = new Server(threadPool);
    ServerConnector connector;

    if (conf.isJobServerSslEnabled()) {
      LOG.debug("Enabling SSL for submarine job server on port "
          + conf.getServerSslPort());
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
          jobServer,
          new SslConnectionFactory(getSslContextFactory(conf),
              HttpVersion.HTTP_1_1.asString()),
          new HttpConnectionFactory(httpsConfig));
    } else {
      connector = new ServerConnector(jobServer);
    }

    configureRequestHeaderSize(conf, connector);
    // Set some timeout options to make debugging easier.
    int timeout = 1000 * 30;
    connector.setIdleTimeout(timeout);
    connector.setSoLingerTime(-1);
    connector.setHost(conf.getJobServerAddress());
    if (conf.useSsl()) {
      connector.setPort(conf.getJobServerSslPort());
    } else {
      connector.setPort(conf.getJobServerPort());
    }

    jobServer.addConnector(connector);
    return jobServer;
  }

  private static SslContextFactory getSslContextFactory(
      SubmarineConfiguration conf) {
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
        (HttpConnectionFactory) connector
            .getConnectionFactory(HttpVersion.HTTP_1_1.toString());
    int requestHeaderSize = conf.getJettyRequestHeaderSize();
    cf.getHttpConfiguration().setRequestHeaderSize(requestHeaderSize);
  }

}
