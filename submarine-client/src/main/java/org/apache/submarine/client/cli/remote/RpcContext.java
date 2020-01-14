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
import org.apache.submarine.commons.utils.SubmarineConfVars;
import org.apache.submarine.commons.utils.SubmarineConfiguration;

import java.util.HashMap;
import java.util.Map;

public class RpcContext {

  private Map<String, String> submarineJobConfigMap = new HashMap<>();

  public static RpcContext convertClientContextToRpcContext(
      ClientContext clientContext) {
     SubmarineConfiguration submarineConfig =
         clientContext.getSubmarineConfig();
    RpcContext rpcContext = new RpcContext();
    rpcContext.addSubmarineJobConfiguration(rpcContext, submarineConfig);
    return rpcContext;
  }

  private void addSubmarineJobConfiguration(RpcContext rpcContext,
      SubmarineConfiguration submarineConfig) {
    rpcContext.getSubmarineJobConfigMap().put(
        SubmarineConfVars.ConfVars.SUBMARINE_RUNTIME_CLASS.getVarName(),
        submarineConfig.getString(
            SubmarineConfVars.ConfVars.SUBMARINE_RUNTIME_CLASS)
    );
  }

  public Map<String, String> getSubmarineJobConfigMap() {
    return submarineJobConfigMap;
  }

  public RpcContext setSubmarineJobConfigMap(
      Map<String, String> submarineJobConfigMap) {
    this.submarineJobConfigMap = submarineJobConfigMap;
    return this;
  }

}