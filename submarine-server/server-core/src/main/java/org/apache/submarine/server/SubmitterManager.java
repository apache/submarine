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

import org.apache.submarine.commons.utils.SubmarineConfiguration;
import org.apache.submarine.server.api.JobSubmitter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.lang.reflect.Constructor;
import java.net.MalformedURLException;
import java.net.URL;
import java.net.URLClassLoader;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;

/**
 * Submitter Manager, load all the submitter plugin, configured by
 * {@link SubmarineConfiguration.ConfVars#SUBMARINE_SUBMITTERS} and related key.
 */
public class SubmitterManager {
  private static final Logger LOG = LoggerFactory.getLogger(SubmitterManager.class);

  private SubmarineConfiguration conf;
  private ConcurrentMap<String, JobSubmitter> submitterMap = new ConcurrentHashMap<>();

  public SubmitterManager(SubmarineConfiguration conf) {
    this.conf = conf;
    loadSubmitters();
  }

  private void loadSubmitters() {
    LOG.info("Start load submitter plugins...");
    List<String> list = conf.listSubmitter();
    for (String name : list) {
      String clazzName = conf.getSubmitterClass(name);
      String classpath = conf.getSubmitterClassPath(name);
      try {
        ClassLoader classLoader = new URLClassLoader(constructUrlsFromClasspath(classpath));
        Class<?> clazz = Class.forName(clazzName, true, classLoader);
        Class<? extends JobSubmitter> sClass = clazz.asSubclass(JobSubmitter.class);
        Constructor<? extends JobSubmitter> method = sClass.getDeclaredConstructor();
        JobSubmitter submitter = method.newInstance();
        submitter.initialize(conf);
        submitterMap.put(submitter.getSubmitterType(), submitter);
      } catch (Exception e) {
        LOG.error(e.toString(), e);
      }
    }
    LOG.info("Success loaded {} submitters.", submitterMap.size());
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

  /**
   * Get the specified submitter by submitter type
   * @return submitter
   */
  public JobSubmitter getSubmitterByType(String type) {
    return submitterMap.get(type);
  }
}
