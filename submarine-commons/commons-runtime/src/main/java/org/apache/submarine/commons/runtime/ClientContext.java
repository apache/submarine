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

package org.apache.submarine.commons.runtime;

import org.apache.submarine.commons.utils.SubmarineConfiguration;
import org.apache.submarine.commons.runtime.fs.DefaultRemoteDirectoryManager;
import org.apache.submarine.commons.runtime.fs.RemoteDirectoryManager;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.yarn.client.api.YarnClient;
import org.apache.hadoop.yarn.conf.YarnConfiguration;

public class ClientContext {
  private Configuration yarnConf = new YarnConfiguration();

  private volatile RemoteDirectoryManager remoteDirectoryManager;
  private YarnClient yarnClient;
  private SubmarineConfiguration submarineConfig;
  private RuntimeFactory runtimeFactory;

  public ClientContext() {
    submarineConfig = SubmarineConfiguration.getInstance();
  }

  public synchronized YarnClient getOrCreateYarnClient() {
    if (yarnClient == null) {
      yarnClient = YarnClient.createYarnClient();
      yarnClient.init(yarnConf);
      yarnClient.start();
    }
    return yarnClient;
  }

  public Configuration getYarnConfig() {
    return yarnConf;
  }

  public void setYarnConfig(Configuration conf) {
    this.yarnConf = conf;
  }

  public RemoteDirectoryManager getRemoteDirectoryManager() {
    if (remoteDirectoryManager == null) {
      synchronized (this) {
        if (remoteDirectoryManager == null) {
          remoteDirectoryManager = new DefaultRemoteDirectoryManager(this);
        }
      }
    }
    return remoteDirectoryManager;
  }

  public SubmarineConfiguration getSubmarineConfig() {
    return submarineConfig;
  }

  public void setSubmarineConfig(SubmarineConfiguration submarineConfig) {
    this.submarineConfig = submarineConfig;
  }

  public RuntimeFactory getRuntimeFactory() {
    return runtimeFactory;
  }

  public void setRuntimeFactory(RuntimeFactory runtimeFactory) {
    this.runtimeFactory = runtimeFactory;
  }
}
