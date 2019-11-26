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
package org.apache.submarine.interpreter;

import org.apache.commons.lang.StringUtils;
import org.apache.submarine.commons.cluster.ClusterClient;
import org.apache.submarine.commons.cluster.meta.ClusterMeta;
import org.apache.submarine.commons.utils.SubmarineConfiguration;
import org.apache.zeppelin.interpreter.remote.RemoteInterpreterUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import sun.misc.Signal;

import java.io.IOException;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.net.SocketException;
import java.net.URLClassLoader;
import java.net.UnknownHostException;
import java.time.LocalDateTime;
import java.util.HashMap;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.apache.submarine.commons.cluster.meta.ClusterMetaType.INTP_PROCESS_META;

public class InterpreterProcess extends Thread {

  private static Logger LOG = LoggerFactory.getLogger(InterpreterProcess.class);

  private Interpreter interpreter;

  private static InterpreterProcess process;

  private AtomicBoolean isRunning = new AtomicBoolean(false);

  // cluster manager client
  private ClusterClient clusterClient = ClusterClient.getInstance();

  private SubmarineConfiguration sconf = SubmarineConfiguration.getInstance();

  protected String interpreterId;

  public InterpreterProcess() { }

  public InterpreterProcess(String interpreterType, String interpreterId, Boolean onlyTest)
          throws IOException {
    this.interpreterId = interpreterId;
    loadInterpreterPlugin(interpreterType);

    if (onlyTest) {
      boolean testResult = interpreter.test();
      LOG.info("Interpreter test result: {}", testResult);
      System.exit(0);
      return;
    }

    this.clusterClient.start(interpreterId);
    putClusterMeta();
  }

  // Submit interpreter process metadata information to cluster metadata
  private void putClusterMeta() {
    if (!sconf.isClusterMode()){
      return;
    }
    String nodeName = clusterClient.getClusterNodeName();
    String host = null;
    try {
      host = RemoteInterpreterUtils.findAvailableHostAddress();
    } catch (UnknownHostException | SocketException e) {
      LOG.error(e.getMessage(), e);
    }
    // commit interpreter meta
    HashMap<String, Object> meta = new HashMap<>();
    meta.put(ClusterMeta.NODE_NAME, nodeName);
    meta.put(ClusterMeta.INTP_PROCESS_NAME, this.interpreterId);
    meta.put(ClusterMeta.INTP_TSERVER_HOST, host);
    meta.put(ClusterMeta.INTP_START_TIME, LocalDateTime.now());
    meta.put(ClusterMeta.LATEST_HEARTBEAT, LocalDateTime.now());
    meta.put(ClusterMeta.STATUS, ClusterMeta.ONLINE_STATUS);

    clusterClient.putClusterMeta(INTP_PROCESS_META, this.interpreterId, meta);
  }

  // get super interpreter class name
  private String getSuperInterpreterClassName(String intpName) {
    String superIntpClassName = "";
    if (StringUtils.equals(intpName, "python")) {
      superIntpClassName = "org.apache.submarine.interpreter.PythonInterpreter";
    } else if (StringUtils.equals(intpName, "spark")) {
      superIntpClassName = "org.apache.submarine.interpreter.SparkInterpreter";
    } else if (StringUtils.equals(intpName, "sparksql")) {
      superIntpClassName = "org.apache.submarine.interpreter.SparkSqlInterpreter";
    } else {
      throw new RuntimeException("cannot recognize the interpreter: " + intpName);
    }

    return superIntpClassName;
  }

  public synchronized void loadInterpreterPlugin(String pluginName)
          throws IOException {

    LOG.info("Loading Plug name: {}", pluginName);
    String pluginClassName = getSuperInterpreterClassName(pluginName);
    LOG.info("Loading Plug Class name: {}", pluginClassName);

    // load plugin from classpath directly first for these builtin Interpreter Plugin.
    this.interpreter = loadPluginFromClassPath(pluginClassName, null);
    if (this.interpreter == null) {
      throw new IOException("Fail to load plugin: " + pluginName);
    }
  }

  private Interpreter loadPluginFromClassPath(
          String pluginClassName, URLClassLoader pluginClassLoader) {
    Interpreter intpProcess = null;
    try {
      Class<?> clazz = null;
      if (null == pluginClassLoader) {
        clazz = Class.forName(pluginClassName);
      } else {
        clazz = Class.forName(pluginClassName, true, pluginClassLoader);
      }

      Constructor<?> cons[] = clazz.getConstructors();
      for (Constructor<?> constructor : cons) {
        LOG.debug(constructor.getName());
      }

      Method[] methods = clazz.getDeclaredMethods();
      //Loop through the methods and print out their names
      for (Method method : methods) {
        LOG.debug(method.getName());
      }

      intpProcess = (Interpreter) (clazz.getConstructor().newInstance());
      return intpProcess;
    } catch (InstantiationException | IllegalAccessException | ClassNotFoundException
            | NoSuchMethodException | InvocationTargetException e) {
      LOG.warn("Fail to instantiate InterpreterLauncher from classpath directly:", e);
    }

    return intpProcess;
  }

  public boolean isRunning() {
    return isRunning.get();
  }


  public void shutdown() {
    isRunning.set(false);
  }


  public static void main(String[] args) throws InterruptedException, IOException {
    String interpreterType = args[0];
    String interpreterId = args[1];
    boolean onlyTest = false;
    if (args.length == 3 && StringUtils.equals(args[2], "test")) {
      onlyTest = true;
    }

    InterpreterProcess interpreterProcess = new InterpreterProcess(interpreterType, interpreterId, onlyTest);
    interpreterProcess.start();

    // add signal handler
    Signal.handle(new Signal("TERM"), signal -> {
      // clean
      LOG.info("handle signal:{}", signal);
    });

    interpreterProcess.join();
    System.exit(0);
  }

  @Override
  public void run() {
    isRunning.set(true);
    while (isRunning.get()) {
      try {
        // TODO(Xun Liu): Next PR will add Thrift Server in here
        LOG.info("Mock TServer run ...");
        sleep(1000);
      } catch (InterruptedException e) {
        LOG.error(e.getMessage(), e);
      }
    }
  }
}
