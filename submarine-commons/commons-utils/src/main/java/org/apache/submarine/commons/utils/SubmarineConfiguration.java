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
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.StringTokenizer;

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
    return getString(ConfVars.SUBMARINE_SERVER_ADDR);
  }

  public boolean useSsl() {
    return getBoolean(ConfVars.SUBMARINE_SERVER_SSL);
  }

  public int getServerPort() {
    return getInt(ConfVars.SUBMARINE_SERVER_PORT);
  }

  @VisibleForTesting
  public void setServerPort(int port) {
    properties.put(ConfVars.SUBMARINE_SERVER_PORT.getVarName(), String.valueOf(port));
  }

  public int getServerSslPort() {
    return getInt(ConfVars.SUBMARINE_SERVER_SSL_PORT);
  }

  public String getKeyStorePath() {
    String path = getString(ConfVars.SUBMARINE_SERVER_SSL_KEYSTORE_PATH);
    return path;
  }

  public String getKeyStoreType() {
    return getString(ConfVars.SUBMARINE_SERVER_SSL_KEYSTORE_TYPE);
  }

  public String getKeyStorePassword() {
    return getString(ConfVars.SUBMARINE_SERVER_SSL_KEYSTORE_PASSWORD);
  }

  public String getKeyManagerPassword() {
    String password = getString(ConfVars.SUBMARINE_SERVER_SSL_KEY_MANAGER_PASSWORD);
    if (password == null) {
      return getKeyStorePassword();
    } else {
      return password;
    }
  }

  public boolean useClientAuth() {
    return getBoolean(ConfVars.SUBMARINE_SERVER_SSL_CLIENT_AUTH);
  }

  public String getTrustStorePath() {
    String path = getString(ConfVars.SUBMARINE_SERVER_SSL_TRUSTSTORE_PATH);
    if (path == null) {
      path = getKeyStorePath();
    }
    return path;
  }

  public String getTrustStoreType() {
    String type = getString(ConfVars.SUBMARINE_SERVER_SSL_TRUSTSTORE_TYPE);
    if (type == null) {
      return getKeyStoreType();
    } else {
      return type;
    }
  }

  public String getTrustStorePassword() {
    String password = getString(ConfVars.SUBMARINE_SERVER_SSL_TRUSTSTORE_PASSWORD);
    if (password == null) {
      return getKeyStorePassword();
    } else {
      return password;
    }
  }

  public Integer getJettyRequestHeaderSize() {
    return getInt(ConfVars.SUBMARINE_SERVER_JETTY_REQUEST_HEADER_SIZE);
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

  public String getMetastoreJdbcUrl() {
    return getString(ConfVars.METASTORE_JDBC_URL);
  }

  @VisibleForTesting
  public void setMetastoreJdbcUrl(String testMetastoreJdbcUrl) {
    properties.put(ConfVars.METASTORE_JDBC_URL.getVarName(), testMetastoreJdbcUrl);
  }


  @VisibleForTesting
  public void setJdbcUrl(String testJdbcUrl) {
    properties.put(ConfVars.JDBC_URL.getVarName(), testJdbcUrl);
  }

  public String getJdbcUserName() {
    return getString(ConfVars.JDBC_USERNAME);
  }

  public String getMetastoreJdbcUserName() {
    return getString(ConfVars.METASTORE_JDBC_USERNAME);
  }

  @VisibleForTesting
  public void setJdbcUserName(String userName) {
    properties.put(ConfVars.JDBC_USERNAME.getVarName(), userName);
  }

  @VisibleForTesting
  public void setMetastoreJdbcUserName(String metastoreUserName) {
    properties.put(ConfVars.METASTORE_JDBC_USERNAME.getVarName(), metastoreUserName);
  }

  public String getJdbcPassword() {
    return getString(ConfVars.JDBC_PASSWORD);
  }

  public String getMetastoreJdbcPassword() {
    return getString(ConfVars.METASTORE_JDBC_PASSWORD);
  }

  @VisibleForTesting
  public void setJdbcPassword(String password) {
    properties.put(ConfVars.JDBC_PASSWORD.getVarName(), password);
  }

  @VisibleForTesting
  public void setMetastoreJdbcPassword(String metastorePassword) {
    properties.put(ConfVars.METASTORE_JDBC_PASSWORD.getVarName(), metastorePassword);
  }

  public String getClusterAddress() {
    return getString(ConfVars.SUBMARINE_CLUSTER_ADDR);
  }

  public void setClusterAddress(String clusterAddr) {
    properties.put(ConfVars.SUBMARINE_CLUSTER_ADDR.getVarName(), clusterAddr);
  }

  public boolean isClusterMode() {
    String clusterAddr = getString(ConfVars.SUBMARINE_CLUSTER_ADDR);
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

  /**
   * Get all submitters from configuration file
   * @return list
   */
  public List<String> listSubmitter() {
    List<String> values = new ArrayList<>();

    String submitters = getString(ConfVars.SUBMARINE_SUBMITTERS.getVarName());
    if (submitters != null) {
      final String delim = ",";
      StringTokenizer tokenizer = new StringTokenizer(submitters, delim);
      while (tokenizer.hasMoreTokens()) {
        values.add(tokenizer.nextToken());
      }
    }

    return values;
  }

  /**
   * Get the entry class name by the specified name
   * @param name the submitter's name
   * @return class name
   */
  public String getSubmitterEntry(String name) {
    return getString(String.format(ConfVars.SUBMARINE_SUBMITTERS_ENTRY.getVarName(), name));
  }

  /**
   * Get the submitter's classpath by the specified name
   * @param name the submitter's name
   * @return classpath
   */
  public String getSubmitterClassPath(String name) {
    return getString(String.format(ConfVars.SUBMARINE_SUBMITTERS_CLASSPATH.getVarName(), name));
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

  public void updateConfiguration(String name, String value) {
    properties.put(name, value);
  }

  public enum ConfVars {
    SUBMARINE_CONF_DIR("submarine.conf.dir", "conf"),
    SUBMARINE_LOCALIZATION_MAX_ALLOWED_FILE_SIZE_MB(
        "submarine.localization.max-allowed-file-size-mb", 2048L),
    SUBMARINE_SERVER_ADDR("submarine.server.addr", "0.0.0.0"),
    SUBMARINE_SERVER_PORT("submarine.server.port", 8080),
    SUBMARINE_SERVER_SSL("submarine.server.ssl", false),
    SUBMARINE_SERVER_SSL_PORT("submarine.server.ssl.port", 8443),
    SUBMARINE_SERVER_JETTY_THREAD_POOL_MAX("submarine.server.jetty.thread.pool.max", 400),
    SUBMARINE_SERVER_JETTY_THREAD_POOL_MIN("submarine.server.jetty.thread.pool.min", 8),
    SUBMARINE_SERVER_JETTY_THREAD_POOL_TIMEOUT("submarine.server.jetty.thread.pool.timeout", 30),
    SUBMARINE_SERVER_JETTY_REQUEST_HEADER_SIZE("submarine.server.jetty.request.header.size", 8192),
    SUBMARINE_SERVER_SSL_CLIENT_AUTH("submarine.server.ssl.client.auth", false),
    SUBMARINE_SERVER_SSL_KEYSTORE_PATH("submarine.server.ssl.keystore.path", "keystore"),
    SUBMARINE_SERVER_SSL_KEYSTORE_TYPE("submarine.server.ssl.keystore.type", "JKS"),
    SUBMARINE_SERVER_SSL_KEYSTORE_PASSWORD("submarine.server.ssl.keystore.password", ""),
    SUBMARINE_SERVER_SSL_KEY_MANAGER_PASSWORD("submarine.server.ssl.key.manager.password", null),
    SUBMARINE_SERVER_SSL_TRUSTSTORE_PATH("submarine.server.ssl.truststore.path", null),
    SUBMARINE_SERVER_SSL_TRUSTSTORE_TYPE("submarine.server.ssl.truststore.type", null),
    SUBMARINE_SERVER_SSL_TRUSTSTORE_PASSWORD("submarine.server.ssl.truststore.password", null),
    SUBMARINE_CLUSTER_ADDR("submarine.cluster.addr", ""),
    SUBMARINE_SERVER_REMOTE_EXECUTION_ENABLED(
        "submarine.server.remote.execution.enabled", false),
    SUBMARINE_SERVER_REMOTE_EXECUTION_PORT(
        "submarine.server.remote.execution.port", 8980),
    CLUSTER_HEARTBEAT_INTERVAL("cluster.heartbeat.interval", 3000),
    CLUSTER_HEARTBEAT_TIMEOUT("cluster.heartbeat.timeout", 9000),
    JDBC_DRIVERCLASSNAME("jdbc.driverClassName", "com.mysql.jdbc.Driver"),
    JDBC_URL("jdbc.url", "jdbc:mysql://127.0.0.1:3306/submarine" +
        "?useUnicode=true&amp;characterEncoding=UTF-8&amp;autoReconnect=true&amp;" +
        "failOverReadOnly=false&amp;zeroDateTimeBehavior=convertToNull&amp;useSSL=false"),

    JDBC_USERNAME("jdbc.username", "submarine"),
    JDBC_PASSWORD("jdbc.password", "password"),
    METASTORE_JDBC_URL("metastore.jdbc.url", "jdbc:mysql://127.0.0.1:3306/metastore" +
        "?useUnicode=true&amp;characterEncoding=UTF-8&amp;autoReconnect=true&amp;" +
        "failOverReadOnly=false&amp;zeroDateTimeBehavior=convertToNull&amp;useSSL=false"),
    METASTORE_JDBC_USERNAME("metastore.jdbc.username", "metastore"),
    METASTORE_JDBC_PASSWORD("metastore.jdbc.password", "password"),
    WORKBENCH_WEBSOCKET_MAX_TEXT_MESSAGE_SIZE(
        "workbench.websocket.max.text.message.size", "1024000"),
    WORKBENCH_WEB_WAR("workbench.web.war", "submarine-workbench/workbench-web/dist"),
    SUBMARINE_RUNTIME_CLASS("submarine.runtime.class",
        "org.apache.submarine.server.submitter.yarn.YarnRuntimeFactory"),
    SUBMARINE_SUBMITTERS("submarine.submitters", ""),
    SUBMARINE_SUBMITTERS_ENTRY("submarine.submitters.*.class", ""),
    SUBMARINE_SUBMITTERS_CLASSPATH("submarine.submitters.*.classpath", "");

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
