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
import org.apache.zeppelin.interpreter.InterpreterContext;
import org.apache.zeppelin.interpreter.InterpreterOutput;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import sun.misc.Signal;
import sun.misc.SignalHandler;

import java.io.File;
import java.io.IOException;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.net.URL;
import java.net.URLClassLoader;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;

/**
 * Entry point for Submarine Interpreter process.
 * Accepting thrift connections from Submarine Workbench Server.
 */
public abstract class InterpreterProcess {
  public abstract void open();
  public abstract InterpreterResult interpret(String code);
  public abstract void close();
  public abstract void cancel();
  public abstract int getProgress();
  public abstract boolean test();

  private static final Logger LOG = LoggerFactory.getLogger(InterpreterProcess.class);

  protected static String interpreterId;

  public static void main(String[] args) throws InterruptedException, IOException {
    String interpreterName = args[0];
    String interpreterId = args[1];
    String onlyTest = "";
    if (args.length == 3) {
      onlyTest = args[2];
    }

    InterpreterProcess interpreterProcess = loadInterpreterPlugin(interpreterName);

    if (StringUtils.equals(onlyTest, "test")) {
      boolean testResult = interpreterProcess.test();
      LOG.info("Interpreter test result: {}", testResult);
      System.exit(0);
      return;
    }

    // add signal handler
    Signal.handle(new Signal("TERM"), new SignalHandler() {
      @Override
      public void handle(Signal signal) {
        // clean
        LOG.info("handle signal:{}", signal);
      }
    });

    System.exit(0);
  }

  // get super interpreter class name
  private static String getSuperInterpreterClassName(String intpName) {
    String superIntpClassName = "";
    if (StringUtils.equals(intpName, "python")) {
      superIntpClassName = "org.apache.submarine.interpreter.PythonInterpreter";
    } else {
      LOG.error("Error interpreter name : {}!", intpName);
    }

    return superIntpClassName;
  }

  public static synchronized InterpreterProcess loadInterpreterPlugin(String pluginName)
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

  private static InterpreterProcess loadPluginFromClassPath(
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
  private static URLClassLoader getPluginClassLoader(String pluginsDir, String pluginName)
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

  protected static InterpreterContext getIntpContext() {
    return InterpreterContext.builder()
        .setInterpreterOut(new InterpreterOutput(null))
        .build();
  }
}
