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
import org.apache.zeppelin.interpreter.InterpreterContext;
import org.apache.zeppelin.interpreter.InterpreterOutput;
import org.apache.zeppelin.interpreter.remote.RemoteInterpreterUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import sun.misc.Signal;
import sun.misc.SignalHandler;

import java.io.File;
import java.io.IOException;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.net.SocketException;
import java.net.URL;
import java.net.URLClassLoader;
import java.net.UnknownHostException;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Properties;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.apache.submarine.commons.cluster.meta.ClusterMetaType.INTP_PROCESS_META;

/**
 * Entry point for Submarine Interpreter process.
 * Accepting thrift connections from Submarine Workbench Server.
 */
public class InterpreterProcess extends Thread implements Interpreter {
  private static final Logger LOG = LoggerFactory.getLogger(InterpreterProcess.class);

  // cluster manager client
  private ClusterClient clusterClient = ClusterClient.getInstance();

  private SubmarineConfiguration sconf = SubmarineConfiguration.getInstance();

  protected String interpreterId;

  private InterpreterProcess interpreterProcess;

  private AtomicBoolean isRunning = new AtomicBoolean(false);

  public static void main(String[] args) throws InterruptedException, IOException {
    String interpreterType = args[0];
    String interpreterId = args[1];
    Boolean onlyTest = false;
    if (args.length == 3 && StringUtils.equals(args[2], "test")) {
      onlyTest = true;
    }

    InterpreterProcess interpreterProcess = new InterpreterProcess(interpreterType, interpreterId, onlyTest);
    interpreterProcess.start();

    // add signal handler
    Signal.handle(new Signal("TERM"), new SignalHandler() {
      @Override
      public void handle(Signal signal) {
        // clean
        LOG.info("handle signal:{}", signal);
      }
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

  public boolean isRunning() {
    return isRunning.get();
  }

  public InterpreterProcess() { }

  public InterpreterProcess(String interpreterType, String interpreterId, Boolean onlyTest)
      throws IOException {
    this.interpreterId = interpreterId;
    this.interpreterProcess = loadInterpreterPlugin(interpreterType);

    if (true == onlyTest) {
      boolean testResult = interpreterProcess.test();
      LOG.info("Interpreter test result: {}", testResult);
      System.exit(0);
      return;
    }

    this.clusterClient.start(interpreterId);
    putClusterMeta();
  }

  // Submit interpreter process metadata information to cluster metadata
  private void putClusterMeta() {
    if (!sconf.workbenchIsClusterMode()){
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
    } else {
      superIntpClassName = "org.apache.submarine.interpreter.InterpreterProcess";
    }

    return superIntpClassName;
  }

  public synchronized InterpreterProcess loadInterpreterPlugin(String pluginName)
      throws IOException {

    LOG.info("Loading Plug name: {}", pluginName);
    String pluginClassName = getSuperInterpreterClassName(pluginName);
    LOG.info("Loading Plug Class name: {}", pluginClassName);

    // load plugin from classpath directly first for these builtin Interpreter Plugin.
    InterpreterProcess intpProcess = loadPluginFromClassPath(pluginClassName, null);
    if (intpProcess == null) {
      throw new IOException("Fail to load plugin: " + pluginName);
    }

    return intpProcess;
  }

  private InterpreterProcess loadPluginFromClassPath(
      String pluginClassName, URLClassLoader pluginClassLoader) {
    InterpreterProcess intpProcess = null;
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

      intpProcess = (InterpreterProcess) (clazz.getConstructor().newInstance());
      return intpProcess;
    } catch (InstantiationException | IllegalAccessException | ClassNotFoundException
        | NoSuchMethodException | InvocationTargetException e) {
      LOG.warn("Fail to instantiate InterpreterLauncher from classpath directly:", e);
    }

    return intpProcess;
  }

  // Get the class load from the specified path
  private URLClassLoader getPluginClassLoader(String pluginsDir, String pluginName)
      throws IOException {
    File pluginFolder = new File(pluginsDir + "/" + pluginName);
    if (!pluginFolder.exists() || pluginFolder.isFile()) {
      LOG.warn("PluginFolder {} doesn't exist or is not a directory", pluginFolder.getAbsolutePath());
      return null;
    }
    List<URL> urls = new ArrayList<>();
    for (File file : pluginFolder.listFiles()) {
      LOG.debug("Add file {} to classpath of plugin: {}", file.getAbsolutePath(), pluginName);
      urls.add(file.toURI().toURL());
    }
    if (urls.isEmpty()) {
      LOG.warn("Can not load plugin {}, because the plugin folder {} is empty.", pluginName, pluginFolder);
      return null;
    }
    return new URLClassLoader(urls.toArray(new URL[0]));
  }

  protected Properties mergeZeplPyIntpProp(Properties newProps) {
    Properties properties = new Properties();
    // Max number of dataframe rows to display.
    properties.setProperty("zeppelin.python.maxResult", "1000");
    // whether use IPython when it is available
    properties.setProperty("zeppelin.python.useIPython", "false");
    properties.setProperty("zeppelin.python.gatewayserver_address", "127.0.0.1");

    if (null != newProps) {
      newProps.putAll(properties);
      return newProps;
    } else {
      return properties;
    }
  }

  protected Properties mergeZeplSparkIntpProp(Properties newProps) {
    Properties properties = new Properties();

    properties.setProperty("zeppelin.spark.maxResult", "1000");
    properties.setProperty("zeppelin.spark.scala.color", "false");

    if (null != newProps) {
      newProps.putAll(properties);
      return newProps;
    } else {
      return properties;
    }
  }

  protected static InterpreterContext getIntpContext() {
    return InterpreterContext.builder()
        .setInterpreterOut(new InterpreterOutput(null))
        .build();
  }

  @Override
  public void shutdown() {
    isRunning.set(false);
  }

  @Override
  public void open() {
    LOG.error("Please implement the open() method of the child class!");
  }

  @Override
  public InterpreterResult interpret(String code) {
    LOG.error("Please implement the interpret() method of the child class!");
    return null;
  }

  @Override
  public void close() {
    LOG.error("Please implement the close() method of the child class!");
  }

  @Override
  public void cancel() {
    LOG.error("Please implement the cancel() method of the child class!");
  }

  @Override
  public int getProgress() {
    LOG.error("Please implement the getProgress() method of the child class!");
    return 0;
  }

  @Override
  public boolean test() {
    LOG.error("Please implement the test() method of the child class!");
    return false;
  }
}
