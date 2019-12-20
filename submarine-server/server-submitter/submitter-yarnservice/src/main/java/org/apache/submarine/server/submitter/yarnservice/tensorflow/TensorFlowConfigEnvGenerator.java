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

package org.apache.submarine.server.submitter.yarnservice.tensorflow;

import org.apache.submarine.commons.runtime.conf.Envs;
import org.apache.submarine.server.submitter.yarnservice.YarnServiceUtils;

public class TensorFlowConfigEnvGenerator {

  public static String getTFConfigEnv(String componentName, int nWorkers,
      int nPs, String serviceName, String userName, String domain) {
    String commonEndpointSuffix = YarnServiceUtils
        .getDNSNameCommonSuffix(serviceName, userName, domain, 8000);

    TFConfigEnv tfConfigEnv =
        new TFConfigEnv(nWorkers, nPs, componentName, commonEndpointSuffix);
    return tfConfigEnv.toJson();
  }

  private static class TFConfigEnv {
    private final int nWorkers;
    private final int nPS;
    private final String componentName;
    private final String endpointSuffix;

    TFConfigEnv(int nWorkers, int nPS, String componentName,
          String endpointSuffix) {
      this.nWorkers = nWorkers;
      this.nPS = nPS;
      this.componentName = componentName;
      this.endpointSuffix = endpointSuffix;
    }

    // Can't just return standard json string. Because the command,
    // export TF_CONFIG="json", would omit " in json string. " needs to be
    // changed to \"
    String toJson() {
      String json = "{\\\"cluster\\\":{";

      String master = getComponentArrayJson("master", 1, endpointSuffix)
          + ",";
      String worker = getComponentArrayJson("worker", nWorkers - 1,
          endpointSuffix) + ",";
      String ps = getComponentArrayJson("ps", nPS, endpointSuffix) + "},";

      StringBuilder sb = new StringBuilder();
      sb.append("\\\"task\\\":{");
      sb.append(" \\\"type\\\":\\\"");
      sb.append(componentName);
      sb.append("\\\",");
      sb.append(" \\\"index\\\":");
      sb.append('$');
      sb.append(Envs.TASK_INDEX_ENV + "},");
      String task = sb.toString();
      String environment = "\\\"environment\\\":\\\"cloud\\\"}";

      sb = new StringBuilder();
      sb.append(json);
      sb.append(master);
      sb.append(worker);
      sb.append(ps);
      sb.append(task);
      sb.append(environment);
      return sb.toString();
    }

    private String getComponentArrayJson(String componentName, int count,
        String endpointSuffix) {
      String component = "\\\"" + componentName + "\\\":";
      StringBuilder array = new StringBuilder();
      array.append("[");
      for (int i = 0; i < count; i++) {
        array.append("\\\"");
        array.append(componentName);
        array.append("-");
        array.append(i);
        array.append(endpointSuffix);
        array.append("\\\"");
        if (i != count - 1) {
          array.append(",");
        }
      }
      array.append("]");
      return component + array.toString();
    }
  }
}
