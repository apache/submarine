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

package org.apache.submarine.client.cli.remote;

import org.apache.submarine.commons.runtime.ClientContext;
import org.apache.submarine.commons.runtime.JobMonitor;
import org.apache.submarine.commons.runtime.JobSubmitter;
import org.apache.submarine.commons.runtime.RuntimeFactory;
import org.apache.submarine.commons.runtime.fs.MemorySubmarineStorage;
import org.apache.submarine.commons.runtime.fs.SubmarineStorage;
import org.apache.submarine.commons.utils.SubmarineConfVars;

/**
 * Implementation of RuntimeFactory with rpc server
 */
public class RpcRuntimeFactory extends RuntimeFactory {
  private JobSubmitterRpcImpl submitter;

  public RpcRuntimeFactory(ClientContext clientContext) {
    super(clientContext);
    String remoteHost = clientContext.getSubmarineConfig().getServerAddress();
    int port = clientContext.getSubmarineConfig().getInt(
        SubmarineConfVars.ConfVars.SUBMARINE_SERVER_REMOTE_EXECUTION_PORT);
    submitter = new JobSubmitterRpcImpl(remoteHost, port, clientContext);
  }

  @Override
  protected JobSubmitter internalCreateJobSubmitter() {
    return submitter;
  }

  @Override
  protected JobMonitor internalCreateJobMonitor() {
    return null;
  }

  @Override
  protected SubmarineStorage internalCreateSubmarineStorage() {
    return new MemorySubmarineStorage();
  }
}
