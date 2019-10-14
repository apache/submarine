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

package org.apache.submarine.runtimes.tony;

import com.linkedin.tony.TonyClient;
import org.apache.hadoop.conf.Configuration;
import org.apache.submarine.commons.runtime.ClientContext;
import org.apache.submarine.commons.runtime.RuntimeFactory;
import org.apache.submarine.commons.runtime.fs.FSBasedSubmarineStorageImpl;
import org.apache.submarine.commons.runtime.JobMonitor;
import org.apache.submarine.commons.runtime.JobSubmitter;
import org.apache.submarine.commons.runtime.fs.SubmarineStorage;

/**
 * Implementation of RuntimeFactory with Tony Runtime
 */
public class TonyRuntimeFactory extends RuntimeFactory {
  private TonyClient tonyClient;
  private TonyJobSubmitter submitter;
  private TonyJobMonitor monitor;

  public TonyRuntimeFactory(ClientContext clientContext) {
    super(clientContext);
    submitter = new TonyJobSubmitter();
    tonyClient = new TonyClient(submitter, new Configuration());
    monitor = new TonyJobMonitor(clientContext, tonyClient);
    submitter.setTonyClient(tonyClient);
  }

  @Override
  protected JobSubmitter internalCreateJobSubmitter() {
    return submitter;
  }

  @Override
  protected JobMonitor internalCreateJobMonitor() {
    return monitor;
  }

  @Override
  protected SubmarineStorage internalCreateSubmarineStorage() {
    return new FSBasedSubmarineStorageImpl(super.clientContext);
  }
}
