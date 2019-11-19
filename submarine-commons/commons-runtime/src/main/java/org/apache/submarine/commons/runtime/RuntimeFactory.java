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

import com.google.common.annotations.VisibleForTesting;
import org.apache.submarine.commons.utils.SubmarineConfiguration;
import org.apache.submarine.commons.runtime.exception.SubmarineRuntimeException;
import org.apache.submarine.commons.runtime.fs.SubmarineStorage;

import java.lang.reflect.InvocationTargetException;

public abstract class RuntimeFactory {
  protected ClientContext clientContext;
  private JobSubmitter jobSubmitter;
  private JobMonitor jobMonitor;
  private SubmarineStorage submarineStorage;

  public RuntimeFactory(ClientContext clientContext) {
    this.clientContext = clientContext;
  }

  public static RuntimeFactory getRuntimeFactory(
      ClientContext clientContext) {
    SubmarineConfiguration submarineConfiguration =
        clientContext.getSubmarineConfig();
    String runtimeClass = submarineConfiguration.getString(
        SubmarineConfiguration.ConfVars.SUBMARINE_RUNTIME_CLASS);

    try {
      Class<?> runtimeClazz = Class.forName(runtimeClass);
      if (RuntimeFactory.class.isAssignableFrom(runtimeClazz)) {
        return (RuntimeFactory) runtimeClazz.getConstructor(ClientContext.class).newInstance(clientContext);
      } else {
        throw new SubmarineRuntimeException("Class: " + runtimeClass
            + " not instance of " + RuntimeFactory.class.getCanonicalName());
      }
    } catch (ClassNotFoundException | IllegalAccessException |
        InstantiationException | NoSuchMethodException |
        InvocationTargetException e) {
      throw new SubmarineRuntimeException(
          "Could not instantiate RuntimeFactory: " + runtimeClass, e);
    }
  }

  protected abstract JobSubmitter internalCreateJobSubmitter();

  protected abstract JobMonitor internalCreateJobMonitor();

  protected abstract SubmarineStorage internalCreateSubmarineStorage();

  public synchronized JobSubmitter getJobSubmitterInstance() {
    if (jobSubmitter == null) {
      jobSubmitter = internalCreateJobSubmitter();
    }
    return jobSubmitter;
  }

  public synchronized JobMonitor getJobMonitorInstance() {
    if (jobMonitor == null) {
      jobMonitor = internalCreateJobMonitor();
    }
    return jobMonitor;
  }

  public synchronized SubmarineStorage getSubmarineStorage() {
    if (submarineStorage == null) {
      submarineStorage = internalCreateSubmarineStorage();
    }
    return submarineStorage;
  }

  @VisibleForTesting
  public synchronized void setJobSubmitterInstance(JobSubmitter jobSubmitter) {
    this.jobSubmitter = jobSubmitter;
  }

  @VisibleForTesting
  public synchronized void setJobMonitorInstance(JobMonitor jobMonitor) {
    this.jobMonitor = jobMonitor;
  }

  @VisibleForTesting
  public synchronized void setSubmarineStorage(SubmarineStorage storage) {
    this.submarineStorage = storage;
  }
}
