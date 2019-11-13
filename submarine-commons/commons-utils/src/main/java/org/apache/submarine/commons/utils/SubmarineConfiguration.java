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
package org.apache.submarine.commons.utils;

import com.google.common.annotations.VisibleForTesting;
import org.apache.commons.configuration.ConfigurationException;
import org.apache.commons.configuration.XMLConfiguration;
import org.apache.commons.configuration.tree.ConfigurationNode;
import org.apache.commons.lang.StringUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.net.URL;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class SubmarineConfiguration extends XMLConfiguration {
  private static final Logger LOG = LoggerFactory.getLogger(SubmarineConfiguration.class);
  private static final long serialVersionUID = 4749303235693848035L;

  private static final String SUBMARINE_SITE_XML = "submarine-site.xml";

  public static final String SUBMARINE_RUNTIME_APP_TYPE = "SUBMARINE";

  private static SubmarineConfiguration conf;

  private Map<String, String> properties = new HashMap<>();

  private SubmarineConfiguration(URL url) throws ConfigurationException {
    setDelimiterParsingDisabled(true);
    load(url);
    initProperties();
  }

  private void initProperties() {
    List<ConfigurationNode> nodes = getRootNode().getChildren();
    if (nodes == null || nodes.isEmpty()) {
      return;
    }
    for (ConfigurationNode p : nodes) {
      String name = (String) p.getChildren("name").get(0).getValue();
      String value = (String) p.getChildren("value").get(0).getValue();
      if (!StringUtils.isEmpty(name)) {
        properties.put(name, value);
      }
    }
  }

  private SubmarineConfiguration() {
    ConfVars[] vars = ConfVars.values();
    for (ConfVars v : vars) {
      if (v.getType() == ConfVars.VarType.BOOLEAN) {
        this.setProperty(v.getVarName(), v.getBooleanValue());
      } else if (v.getType() == ConfVars.VarType.LONG) {
        this.setProperty(v.getVarName(), v.getLongValue());
      } else if (v.getType() == ConfVars.VarType.INT) {
        this.setProperty(v.getVarName(), v.getIntValue());
      } else if (v.getType() == ConfVars.VarType.FLOAT) {
        this.setProperty(v.getVarName(), v.getFloatValue());
      } else if (v.getType() == ConfVars.VarType.STRING) {
        this.setProperty(v.getVarName(), v.getStringValue());
      } else {
        throw new RuntimeException("Unsupported VarType");
      }
    }
  }

  public static SubmarineConfiguration getInstance() {
    if (conf == null) {
      synchronized (SubmarineConfiguration.class) {
        if  (conf == null) {
          conf = newInstance();
        }
      }
    }
    return conf;
  }

  public static SubmarineConfiguration newInstance() {
    SubmarineConfiguration submarineConfig;
    ClassLoader classLoader = Thread.currentThread().getContextClassLoader();
    URL url;

    url = SubmarineConfiguration.class.getResource(SUBMARINE_SITE_XML);
    if (url == null) {
      ClassLoader cl = SubmarineConfiguration.class.getClassLoader();
      if (cl != null) {
        url = cl.getResource(SUBMARINE_SITE_XML);
      }
    }
    if (url == null) {
      url = classLoader.getResource(SUBMARINE_SITE_XML);
    }

    if (url == null) {
      LOG.warn("Failed to load configuration, proceeding with a default");
      submarineConfig = new SubmarineConfiguration();
    } else {
      try {
        LOG.info("Load configuration from " + url);
        submarineConfig = new SubmarineConfiguration(url);
      } catch (ConfigurationException e) {
        LOG.warn("Failed to load configuration from " + url + " proceeding with a default", e);
        submarineConfig = new SubmarineConfiguration();
      }
    }

    return submarineConfig;
  }

  public String getServerAddress() {
    return getString(ConfVars.SERVER_ADDR);
  }

  public String getJobServerAddress() {
    return getString(ConfVars.JOB_SERVER_ADDR);
  }

  public boolean useSsl() {
    return getBoolean(ConfVars.SERVER_SSL);
  }

  public boolean isJobServerSslEnabled() {
    return getBoolean(ConfVars.JOB_SERVER_SSL);
  }

  public int getServerPort() {
    return getInt(ConfVars.SERVER_PORT);
  }

  public int getJobServerPort() {
    return getInt(ConfVars.JOB_SERVER_PORT);
  }

  @VisibleForTesting
  public void setServerPort(int port) {
    properties.put(ConfVars.SERVER_PORT.getVarName(), String.valueOf(port));
  }

  public int getServerSslPort() {
    return getInt(ConfVars.SERVER_SSL_PORT);
  }

  public int getJobServerSslPort() {
    return getInt(ConfVars.JOB_SERVER_SSL_PORT);
  }

  public String getJobServerUrlPrefix() {
    return getString(ConfVars.JOB_SERVER_REST_URL_PREFIX);
  }

  public String getKeyStorePath() {
    String path = getString(ConfVars.SSL_KEYSTORE_PATH);
    return path;
  }

  public String getKeyStoreType() {
    return getString(ConfVars.SERVER_SSL_KEYSTORE_TYPE);
  }

  public String getKeyStorePassword() {
    return getString(ConfVars.SERVER_SSL_KEYSTORE_PASSWORD);
  }

  public String getKeyManagerPassword() {
    String password = getString(ConfVars.SERVER_SSL_KEY_MANAGER_PASSWORD);
    if (password == null) {
      return getKeyStorePassword();
    } else {
      return password;
    }
  }

  public boolean useClientAuth() {
    return getBoolean(ConfVars.SSL_CLIENT_AUTH);
  }

  public String getTrustStorePath() {
    String path = getString(ConfVars.SERVER_SSL_TRUSTSTORE_PATH);
    if (path == null) {
      path = getKeyStorePath();
    }
    return path;
  }

  public String getTrustStoreType() {
    String type = getString(ConfVars.SERVER_SSL_TRUSTSTORE_TYPE);
    if (type == null) {
      return getKeyStoreType();
    } else {
      return type;
    }
  }

  public String getTrustStorePassword() {
    String password = getString(ConfVars.SERVER_SSL_TRUSTSTORE_PASSWORD);
    if (password == null) {
      return getKeyStorePassword();
    } else {
      return password;
    }
  }

  public Integer getJettyRequestHeaderSize() {
    return getInt(ConfVars.SERVER_JETTY_REQUEST_HEADER_SIZE);
  }

  public String getRelativeDir(ConfVars c) {
    return getRelativeDir(getString(c));
  }

  public String getRelativeDir(String path) {
    if (path != null && path.startsWith(File.separator) || isWindowsPath(path)) {
      return path;
    } else {
      return getString("./") + File.separator + path;
    }
  }

  public boolean isWindowsPath(String path) {
    return path.matches("^[A-Za-z]:\\\\.*");
  }

  public String getJdbcDriverClassName() {
    return getString(ConfVars.JDBC_DRIVERCLASSNAME);
  }

  public String getJdbcUrl() {
    return getString(ConfVars.JDBC_URL);
  }

  @VisibleForTesting
  public void setJdbcUrl(String testJdbcUrl) {
    properties.put(ConfVars.JDBC_URL.getVarName(), testJdbcUrl);
  }

  public String getJdbcUserName() {
    return getString(ConfVars.JDBC_USERNAME);
  }

  @VisibleForTesting
  public void setJdbcUserName(String userName) {
    properties.put(ConfVars.JDBC_USERNAME.getVarName(), userName);
  }

  public String getJdbcPassword() {
    return getString(ConfVars.JDBC_PASSWORD);
  }

  @VisibleForTesting
  public void setJdbcPassword(String password) {
    properties.put(ConfVars.JDBC_PASSWORD.getVarName(), password);
  }

  public String getClusterAddress() {
    return getString(ConfVars.WORKBENCH_CLUSTER_ADDR);
  }

  public void setClusterAddress(String clusterAddr) {
    properties.put(ConfVars.WORKBENCH_CLUSTER_ADDR.getVarName(), clusterAddr);
  }

  public boolean workbenchIsClusterMode() {
    String clusterAddr = getString(ConfVars.WORKBENCH_CLUSTER_ADDR);
    if (StringUtils.isEmpty(clusterAddr)) {
      return false;
    }

    return true;
  }

  public int getClusterHeartbeatInterval() {
    return getInt(ConfVars.CLUSTER_HEARTBEAT_INTERVAL);
  }

  public int getClusterHeartbeatTimeout() {
    return getInt(ConfVars.CLUSTER_HEARTBEAT_TIMEOUT);
  }

  public String getWebsocketMaxTextMessageSize() {
    return getString(ConfVars.WORKBENCH_WEBSOCKET_MAX_TEXT_MESSAGE_SIZE);
  }

  private String getStringValue(String name, String d) {
    String value = this.properties.get(name);
    if (value != null) {
      return value;
    }
    return d;
  }

  private int getIntValue(String name, int d) {
    String value = this.properties.get(name);
    if (value != null) {
      return Integer.parseInt(value);
    }
    return d;
  }

  private long getLongValue(String name, long d) {
    String value = this.properties.get(name);
    if (value != null) {
      return Long.parseLong(value);
    }
    return d;
  }

  private float getFloatValue(String name, float d) {
    String value = this.properties.get(name);
    if (value != null) {
      return Float.parseFloat(value);
    }
    return d;
  }

  private boolean getBooleanValue(String name, boolean d) {
    String value = this.properties.get(name);
    if (value != null) {
      return Boolean.parseBoolean(value);
    }
    return d;
  }

  public String getString(ConfVars c) {
    return getString(c.name(), c.getVarName(), c.getStringValue());
  }

  public String getString(String envName, String propertyName, String defaultValue) {
    if (System.getenv(envName) != null) {
      return System.getenv(envName);
    }
    if (System.getProperty(propertyName) != null) {
      return System.getProperty(propertyName);
    }

    return getStringValue(propertyName, defaultValue);
  }

  public void setString(ConfVars c, String value) {
    properties.put(c.getVarName(), value);
  }

  public int getInt(ConfVars c) {
    return getInt(c.name(), c.getVarName(), c.getIntValue());
  }

  public void setInt(ConfVars c, int value) {
    properties.put(c.getVarName(), String.valueOf(value));
  }

  public int getInt(String envName, String propertyName, int defaultValue) {
    if (System.getenv(envName) != null) {
      return Integer.parseInt(System.getenv(envName));
    }

    if (System.getProperty(propertyName) != null) {
      return Integer.parseInt(System.getProperty(propertyName));
    }
    return getIntValue(propertyName, defaultValue);
  }

  public long getLong(ConfVars c) {
    return getLong(c.name(), c.getVarName(), c.getLongValue());
  }

  public void setLong(ConfVars c, long value) {
    properties.put(c.getVarName(), String.valueOf(value));
  }

  public long getLong(String envName, String propertyName, long defaultValue) {
    if (System.getenv(envName) != null) {
      return Long.parseLong(System.getenv(envName));
    }

    if (System.getProperty(propertyName) != null) {
      return Long.parseLong(System.getProperty(propertyName));
    }
    return getLongValue(propertyName, defaultValue);
  }

  public float getFloat(ConfVars c) {
    return getFloat(c.name(), c.getVarName(), c.getFloatValue());
  }

  public float getFloat(String envName, String propertyName, float defaultValue) {
    if (System.getenv(envName) != null) {
      return Float.parseFloat(System.getenv(envName));
    }
    if (System.getProperty(propertyName) != null) {
      return Float.parseFloat(System.getProperty(propertyName));
    }
    return getFloatValue(propertyName, defaultValue);
  }

  public boolean getBoolean(ConfVars c) {
    return getBoolean(c.name(), c.getVarName(), c.getBooleanValue());
  }

  public boolean getBoolean(String envName, String propertyName, boolean defaultValue) {
    if (System.getenv(envName) != null) {
      return Boolean.parseBoolean(System.getenv(envName));
    }

    if (System.getProperty(propertyName) != null) {
      return Boolean.parseBoolean(System.getProperty(propertyName));
    }
    return getBooleanValue(propertyName, defaultValue);
  }

  public enum ConfVars {
    SUBMARINE_CONF_DIR("submarine.conf.dir", "conf"),
    SUBMARINE_LOCALIZATION_MAX_ALLOWED_FILE_SIZE_MB(
        "submarine.localization.max-allowed-file-size-mb", 2048L),
    SUBMARINE_RUNTIME_CLASS(
        "submarine.runtime.class",
        "org.apache.submarine.server.submitter.yarn.YarnRuntimeFactory"),
    SERVER_ADDR("workbench.server.addr", "0.0.0.0"),
    SERVER_PORT("workbench.server.port", 8080),
    SERVER_SSL("workbench.server.ssl", false),
    SERVER_SSL_PORT("workbench.server.ssl.port", 8443),
    SERVER_JETTY_THREAD_POOL_MAX("workbench.server.jetty.thread.pool.max", 400),
    SERVER_JETTY_THREAD_POOL_MIN("workbench.server.jetty.thread.pool.min", 8),
    SERVER_JETTY_THREAD_POOL_TIMEOUT("workbench.server.jetty.thread.pool.timeout", 30),
    SERVER_JETTY_REQUEST_HEADER_SIZE("workbench.server.jetty.request.header.size", 8192),
    SSL_CLIENT_AUTH("workbench.ssl.client.auth", false),
    SSL_KEYSTORE_PATH("workbench.ssl.keystore.path", "keystore"),
    WORKBENCH_CLUSTER_ADDR("workbench.cluster.addr", ""),
    CLUSTER_HEARTBEAT_INTERVAL("cluster.heartbeat.interval", 3000),
    CLUSTER_HEARTBEAT_TIMEOUT("cluster.heartbeat.timeout", 9000),
    SERVER_SSL_KEYSTORE_TYPE("workbench.ssl.keystore.type", "JKS"),
    SERVER_SSL_KEYSTORE_PASSWORD("workbench.ssl.keystore.password", ""),
    SERVER_SSL_KEY_MANAGER_PASSWORD("workbench.ssl.key.manager.password", null),
    SERVER_SSL_TRUSTSTORE_PATH("workbench.ssl.truststore.path", null),
    SERVER_SSL_TRUSTSTORE_TYPE("workbench.ssl.truststore.type", null),
    SERVER_SSL_TRUSTSTORE_PASSWORD("workbench.ssl.truststore.password", null),
    JDBC_DRIVERCLASSNAME("jdbc.driverClassName", "com.mysql.jdbc.Driver"),
    JDBC_URL("jdbc.url", "jdbc:mysql://127.0.0.1:3306/submarineDB" +
        "?useUnicode=true&amp;characterEncoding=UTF-8&amp;autoReconnect=true&amp;" +
        "failOverReadOnly=false&amp;zeroDateTimeBehavior=convertToNull&amp;useSSL=false"),
    JDBC_USERNAME("jdbc.username", "submarine"),
    JDBC_PASSWORD("jdbc.password", "password"),
    WORKBENCH_WEBSOCKET_MAX_TEXT_MESSAGE_SIZE(
        "workbench.websocket.max.text.message.size", "1024000"),
    WORKBENCH_WEB_WAR("workbench.web.war", "submarine-workbench/workbench-web/dist"),
    // submarine job server settings
    JOB_SERVER_SSL("job.server.ssl", false),
    JOB_SERVER_SSL_PORT("job.server.ssl.port", 8443),
    JOB_SERVER_ADDR("job.server.port", "0.0.0.0"),
    JOB_SERVER_PORT("job.server.port", 8765),
    JOB_SERVER_REST_URL_PREFIX("job.server.rest.prefix", "/*");

    private String varName;
    @SuppressWarnings("rawtypes")
    private Class varClass;
    private String stringValue;
    private VarType type;
    private int intValue;
    private float floatValue;
    private boolean booleanValue;
    private long longValue;


    ConfVars(String varName, String varValue) {
      this.varName = varName;
      this.varClass = String.class;
      this.stringValue = varValue;
      this.intValue = -1;
      this.floatValue = -1;
      this.longValue = -1;
      this.booleanValue = false;
      this.type = VarType.STRING;
    }

    ConfVars(String varName, int intValue) {
      this.varName = varName;
      this.varClass = Integer.class;
      this.stringValue = null;
      this.intValue = intValue;
      this.floatValue = -1;
      this.longValue = -1;
      this.booleanValue = false;
      this.type = VarType.INT;
    }

    ConfVars(String varName, long longValue) {
      this.varName = varName;
      this.varClass = Integer.class;
      this.stringValue = null;
      this.intValue = -1;
      this.floatValue = -1;
      this.longValue = longValue;
      this.booleanValue = false;
      this.type = VarType.LONG;
    }

    ConfVars(String varName, float floatValue) {
      this.varName = varName;
      this.varClass = Float.class;
      this.stringValue = null;
      this.intValue = -1;
      this.longValue = -1;
      this.floatValue = floatValue;
      this.booleanValue = false;
      this.type = VarType.FLOAT;
    }

    ConfVars(String varName, boolean booleanValue) {
      this.varName = varName;
      this.varClass = Boolean.class;
      this.stringValue = null;
      this.intValue = -1;
      this.longValue = -1;
      this.floatValue = -1;
      this.booleanValue = booleanValue;
      this.type = VarType.BOOLEAN;
    }

    public String getVarName() {
      return varName;
    }

    @SuppressWarnings("rawtypes")
    public Class getVarClass() {
      return varClass;
    }

    public int getIntValue() {
      return intValue;
    }

    public long getLongValue() {
      return longValue;
    }

    public float getFloatValue() {
      return floatValue;
    }

    public String getStringValue() {
      return stringValue;
    }

    public boolean getBooleanValue() {
      return booleanValue;
    }

    public VarType getType() {
      return type;
    }

    enum VarType {
      STRING {
        @Override
        void checkType(String value) throws Exception {
        }
      },
      INT {
        @Override
        void checkType(String value) throws Exception {
          Integer.valueOf(value);
        }
      },
      LONG {
        @Override
        void checkType(String value) throws Exception {
          Long.valueOf(value);
        }
      },
      FLOAT {
        @Override
        void checkType(String value) throws Exception {
          Float.valueOf(value);
        }
      },
      BOOLEAN {
        @Override
        void checkType(String value) throws Exception {
          Boolean.valueOf(value);
        }
      };

      boolean isType(String value) {
        try {
          checkType(value);
        } catch (Exception e) {
          LOG.error("Exception in SubmarineConfiguration while isType", e);
          return false;
        }
        return true;
      }

      String typeString() {
        return name().toUpperCase();
      }

      abstract void checkType(String value) throws Exception;
    }
  }
}
