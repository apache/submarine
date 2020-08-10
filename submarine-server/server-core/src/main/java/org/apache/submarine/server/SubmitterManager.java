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

import org.apache.submarine.commons.utils.SubmarineConfVars;
import org.apache.submarine.commons.utils.SubmarineConfiguration;
import org.apache.submarine.server.api.Submitter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.lang.reflect.Constructor;
import java.net.MalformedURLException;
import java.net.URL;
import java.net.URLClassLoader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Submitter Manager is responsible for load {@link Submitter} implements class.
 */
public class SubmitterManager {
  private static final Logger LOG = LoggerFactory.getLogger(SubmitterManager.class);

  private static final SubmitterManager INSTANCE = new SubmitterManager();

  private final Map<String, JobSubmitterConfig> SUBMITTER_CONFIG_MAP = new HashMap<>();

  private Submitter submitter;

  {
    String home = System.getenv("SUBMARINE_HOME");
    LOG.info("Submarine Home: {}", home);
    SUBMITTER_CONFIG_MAP.put("k8s", new JobSubmitterConfig(
        "org.apache.submarine.server.submitter.k8s.K8sSubmitter",
        home + "/lib/submitter/k8s/"));
  }

  public static Submitter loadSubmitter() {
    return INSTANCE.submitter;
  }

  private SubmitterManager() {
    SubmarineConfiguration conf = SubmarineConfiguration.getInstance();
    String type = conf.getString(SubmarineConfVars.ConfVars.SUBMARINE_SUBMITTER);
    String clazzName = SUBMITTER_CONFIG_MAP.get(type).clazzName;
    String classpath = SUBMITTER_CONFIG_MAP.get(type).classpath;
    try {
      ClassLoader classLoader = new URLClassLoader(constructUrlsFromClasspath(classpath));
      Class<?> clazz = Class.forName(clazzName, true, classLoader);
      Class<? extends Submitter> sClass = clazz.asSubclass(Submitter.class);
      Constructor<? extends Submitter> method = sClass.getDeclaredConstructor();
      submitter = method.newInstance();
      submitter.initialize(conf);
    } catch (Exception e) {
      LOG.error("Initialize the submitter failed. " + e.getMessage(), e);
    }
  }

  private URL[] constructUrlsFromClasspath(String classpath) throws MalformedURLException {
    List<URL> urls = new ArrayList<>();
    for (String path : classpath.split(File.pathSeparator)) {
      if (path.endsWith("/*")) {
        path = path.substring(0, path.length() - 2);
      }

      File file = new File(path);
      if (file.isDirectory()) {
        File[] items = file.listFiles();
        if (items != null) {
          for (File item : items) {
            urls.add(item.toURI().toURL());
          }
        }
      } else {
        urls.add(file.toURI().toURL());
      }
    }
    return urls.toArray(new URL[0]);
  }

  private static class JobSubmitterConfig {
    private final String clazzName;
    private final String classpath;

    JobSubmitterConfig(String clazzName, String classpath) {
      this.clazzName = clazzName;
      this.classpath = classpath;
    }
  }
}
