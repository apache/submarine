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

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class SubmarineConfVars {
  private static final Logger LOG = LoggerFactory.getLogger(SubmarineConfVars.class);
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
    SUBMARINE_SERVER_RPC_ENABLED(
        "submarine.server.rpc.enabled", false),
    SUBMARINE_SERVER_RPC_PORT(
        "submarine.server.rpc.port", 8980),
    CLUSTER_HEARTBEAT_INTERVAL("cluster.heartbeat.interval", 3000),
    CLUSTER_HEARTBEAT_TIMEOUT("cluster.heartbeat.timeout", 9000),

    JDBC_DRIVERCLASSNAME("jdbc.driverClassName", "com.mysql.jdbc.Driver"),
    JDBC_URL("jdbc.url", "jdbc:mysql://127.0.0.1:3306/submarine" +
        "?useUnicode=true&characterEncoding=UTF-8&autoReconnect=true&allowMultiQueries=true&" +
        "failOverReadOnly=false&zeroDateTimeBehavior=convertToNull&useSSL=false"),
    JDBC_USERNAME("jdbc.username", "submarine"),
    JDBC_PASSWORD("jdbc.password", "password"),
    METASTORE_JDBC_URL("metastore.jdbc.url", "jdbc:mysql://127.0.0.1:3306/metastore" +
        "?useUnicode=true&characterEncoding=UTF-8&autoReconnect=true&" +
        "failOverReadOnly=false&zeroDateTimeBehavior=convertToNull&useSSL=false"),
    METASTORE_JDBC_USERNAME("metastore.jdbc.username", "metastore"),
    METASTORE_JDBC_PASSWORD("metastore.jdbc.password", "password"),

    /* cookie setting */
    SUBMARINE_COOKIE_HTTP_ONLY("submarine.cookie.http.only", false),
    SUBMARINE_COOKIE_SECURE("submarine.cookie.secure", false),
    SUBMARINE_COOKIE_SAMESITE("submarine.cookie.samesite", ""),

    /* auth */
    SUBMARINE_AUTH_TYPE("submarine.auth.type", "default"),

    WORKBENCH_WEBSOCKET_MAX_TEXT_MESSAGE_SIZE(
        "workbench.websocket.max.text.message.size", "1024000"),
    WORKBENCH_WEB_WAR("workbench.web.war", "submarine-workbench/workbench-web/dist"),
    SUBMARINE_RUNTIME_CLASS("submarine.runtime.class",
        "org.apache.submarine.server.submitter.yarn.YarnRuntimeFactory"),
    SUBMARINE_SUBMITTER("submarine.submitter", "k8s"),
    ENVIRONMENT_CONDA_MIN_VERSION("environment.conda.min.version", "4.0.1"),
    ENVIRONMENT_CONDA_MAX_VERSION("environment.conda.max.version", "4.10.10");

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
