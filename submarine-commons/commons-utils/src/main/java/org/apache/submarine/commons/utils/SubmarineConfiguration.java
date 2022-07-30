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

  private static volatile SubmarineConfiguration conf;

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
    SubmarineConfVars.ConfVars[] vars = SubmarineConfVars.ConfVars.values();
    for (SubmarineConfVars.ConfVars v : vars) {
      if (v.getType() == SubmarineConfVars.ConfVars.VarType.BOOLEAN) {
        this.setProperty(v.getVarName(), v.getBooleanValue());
      } else if (v.getType() == SubmarineConfVars.ConfVars.VarType.LONG) {
        this.setProperty(v.getVarName(), v.getLongValue());
      } else if (v.getType() == SubmarineConfVars.ConfVars.VarType.INT) {
        this.setProperty(v.getVarName(), v.getIntValue());
      } else if (v.getType() == SubmarineConfVars.ConfVars.VarType.FLOAT) {
        this.setProperty(v.getVarName(), v.getFloatValue());
      } else if (v.getType() == SubmarineConfVars.ConfVars.VarType.STRING) {
        this.setProperty(v.getVarName(), v.getStringValue());
      } else {
        throw new RuntimeException("Unsupported VarType");
      }
    }
  }

  // Get a single instance
  // Note: Cannot be mixed with newInstance()
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

  // Create a new instance
  // Note: Cannot be mixed with getInstance()
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
    return getString(SubmarineConfVars.ConfVars.SUBMARINE_SERVER_ADDR);
  }

  public boolean useSsl() {
    return getBoolean(SubmarineConfVars.ConfVars.SUBMARINE_SERVER_SSL);
  }

  public int getServerPort() {
    return getInt(SubmarineConfVars.ConfVars.SUBMARINE_SERVER_PORT);
  }

  @VisibleForTesting
  public void setServerPort(int port) {
    properties.put(SubmarineConfVars.ConfVars.SUBMARINE_SERVER_PORT.getVarName(), String.valueOf(port));
  }

  public int getServerSslPort() {
    return getInt(SubmarineConfVars.ConfVars.SUBMARINE_SERVER_SSL_PORT);
  }

  public String getKeyStorePath() {
    String path = getString(SubmarineConfVars.ConfVars.SUBMARINE_SERVER_SSL_KEYSTORE_PATH);
    return path;
  }

  public String getKeyStoreType() {
    return getString(SubmarineConfVars.ConfVars.SUBMARINE_SERVER_SSL_KEYSTORE_TYPE);
  }

  public String getKeyStorePassword() {
    return getString(SubmarineConfVars.ConfVars.SUBMARINE_SERVER_SSL_KEYSTORE_PASSWORD);
  }

  public String getKeyManagerPassword() {
    String password = getString(SubmarineConfVars.ConfVars.SUBMARINE_SERVER_SSL_KEY_MANAGER_PASSWORD);
    if (password == null) {
      return getKeyStorePassword();
    } else {
      return password;
    }
  }

  public boolean useClientAuth() {
    return getBoolean(SubmarineConfVars.ConfVars.SUBMARINE_SERVER_SSL_CLIENT_AUTH);
  }

  public String getTrustStorePath() {
    String path = getString(SubmarineConfVars.ConfVars.SUBMARINE_SERVER_SSL_TRUSTSTORE_PATH);
    if (path == null) {
      path = getKeyStorePath();
    }
    return path;
  }

  public String getTrustStoreType() {
    String type = getString(SubmarineConfVars.ConfVars.SUBMARINE_SERVER_SSL_TRUSTSTORE_TYPE);
    if (type == null) {
      return getKeyStoreType();
    } else {
      return type;
    }
  }

  public String getTrustStorePassword() {
    String password = getString(SubmarineConfVars.ConfVars.SUBMARINE_SERVER_SSL_TRUSTSTORE_PASSWORD);
    if (password == null) {
      return getKeyStorePassword();
    } else {
      return password;
    }
  }

  public Integer getJettyRequestHeaderSize() {
    return getInt(SubmarineConfVars.ConfVars.SUBMARINE_SERVER_JETTY_REQUEST_HEADER_SIZE);
  }

  public String getRelativeDir(SubmarineConfVars.ConfVars c) {
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
    return getString(SubmarineConfVars.ConfVars.JDBC_DRIVERCLASSNAME);
  }

  public String getJdbcUrl() {
    return getString(SubmarineConfVars.ConfVars.JDBC_URL);
  }

  public String getMetastoreJdbcUrl() {
    return getString(SubmarineConfVars.ConfVars.METASTORE_JDBC_URL);
  }

  @VisibleForTesting
  public void setMetastoreJdbcUrl(String testMetastoreJdbcUrl) {
    properties.put(SubmarineConfVars.ConfVars.METASTORE_JDBC_URL.getVarName(), testMetastoreJdbcUrl);
  }

  @VisibleForTesting
  public void setJdbcDriverClassName(String driverClassName) {
    properties.put(SubmarineConfVars.ConfVars.JDBC_DRIVERCLASSNAME.getVarName(), driverClassName);
  }

  @VisibleForTesting
  public void setJdbcUrl(String testJdbcUrl) {
    properties.put(SubmarineConfVars.ConfVars.JDBC_URL.getVarName(), testJdbcUrl);
  }

  public String getJdbcUserName() {
    return getString(SubmarineConfVars.ConfVars.JDBC_USERNAME);
  }

  public String getMetastoreJdbcUserName() {
    return getString(SubmarineConfVars.ConfVars.METASTORE_JDBC_USERNAME);
  }

  @VisibleForTesting
  public void setJdbcUserName(String userName) {
    properties.put(SubmarineConfVars.ConfVars.JDBC_USERNAME.getVarName(), userName);
  }

  @VisibleForTesting
  public void setMetastoreJdbcUserName(String metastoreUserName) {
    properties.put(SubmarineConfVars.ConfVars.METASTORE_JDBC_USERNAME.getVarName(), metastoreUserName);
  }

  public String getJdbcPassword() {
    return getString(SubmarineConfVars.ConfVars.JDBC_PASSWORD);
  }

  public String getMetastoreJdbcPassword() {
    return getString(SubmarineConfVars.ConfVars.METASTORE_JDBC_PASSWORD);
  }

  @VisibleForTesting
  public void setJdbcPassword(String password) {
    properties.put(SubmarineConfVars.ConfVars.JDBC_PASSWORD.getVarName(), password);
  }

  @VisibleForTesting
  public void setMetastoreJdbcPassword(String metastorePassword) {
    properties.put(SubmarineConfVars.ConfVars.METASTORE_JDBC_PASSWORD.getVarName(), metastorePassword);
  }

  public String getClusterAddress() {
    return getString(SubmarineConfVars.ConfVars.SUBMARINE_CLUSTER_ADDR);
  }

  public void setClusterAddress(String clusterAddr) {
    properties.put(SubmarineConfVars.ConfVars.SUBMARINE_CLUSTER_ADDR.getVarName(), clusterAddr);
  }

  public boolean isClusterMode() {
    String clusterAddr = getString(SubmarineConfVars.ConfVars.SUBMARINE_CLUSTER_ADDR);
    if (StringUtils.isEmpty(clusterAddr)) {
      return false;
    }

    return true;
  }

  public int getClusterHeartbeatInterval() {
    return getInt(SubmarineConfVars.ConfVars.CLUSTER_HEARTBEAT_INTERVAL);
  }

  public int getClusterHeartbeatTimeout() {
    return getInt(SubmarineConfVars.ConfVars.CLUSTER_HEARTBEAT_TIMEOUT);
  }

  public String getWebsocketMaxTextMessageSize() {
    return getString(SubmarineConfVars.ConfVars.WORKBENCH_WEBSOCKET_MAX_TEXT_MESSAGE_SIZE);
  }

  public String getServerServiceName() {
    return getString(SubmarineConfVars.ConfVars.SUBMARINE_SERVER_SERVICE_NAME);
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

  public String getString(SubmarineConfVars.ConfVars c) {
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

  public void setString(SubmarineConfVars.ConfVars c, String value) {
    properties.put(c.getVarName(), value);
  }

  public int getInt(SubmarineConfVars.ConfVars c) {
    return getInt(c.name(), c.getVarName(), c.getIntValue());
  }

  public void setInt(SubmarineConfVars.ConfVars c, int value) {
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

  public long getLong(SubmarineConfVars.ConfVars c) {
    return getLong(c.name(), c.getVarName(), c.getLongValue());
  }

  public void setLong(SubmarineConfVars.ConfVars c, long value) {
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

  public float getFloat(SubmarineConfVars.ConfVars c) {
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

  public boolean getBoolean(SubmarineConfVars.ConfVars c) {
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

  public void updateConfiguration(String name, String value) {
    properties.put(name, value);
  }
}
